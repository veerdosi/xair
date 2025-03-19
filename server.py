import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

# Import XAIR modules
from backend.llm_interface import LLMInterface
from backend.cgrt import CGRTGenerator, CrossGeneration
from backend.cgrt_tree import CGRTTree
from backend.counterfactual import CounterfactualGenerator
from backend.counterfactual_integrator import CounterfactualIntegrator
from backend.impact_analyzer import ImpactAnalyzer
from backend.error_handling import XAIRError, format_error_for_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="XAIR API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 1000
    max_depth: int = 3
    min_probability: float = 0.1

class CounterfactualRequest(BaseModel):
    tree_id: str

class AnalyzeRequest(BaseModel):
    tree_id: str
    counterfactual_ids: Optional[List[str]] = None

# In-memory storage for trees and related data
trees = {}
cross_generations = {}
counterfactuals_store = {}
impact_scores_store = {}

# LLM interface instance
llm = None
cgrt_generator = None
cf_generator = None
cf_integrator = None
impact_analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize the LLM interface and related components"""
    global llm, cgrt_generator, cf_generator, cf_integrator, impact_analyzer
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("API key is required. Set OPENAI_API_KEY environment variable")
        raise RuntimeError("API key is required")
    
    # Initialize LLM interface
    llm = LLMInterface(
        api_key=api_key,
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        temperature=float(os.getenv("TEMPERATURE", "0.7"))
    )
    
    # Initialize tree generator and related components
    cgrt_generator = CGRTGenerator(llm)
    cf_generator = CounterfactualGenerator(llm, cgrt_generator)
    cf_integrator = CounterfactualIntegrator(cgrt_generator, cf_generator)
    impact_analyzer = ImpactAnalyzer(llm, cgrt_generator, cf_generator, cf_integrator)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    if llm:
        await llm.close()

@app.exception_handler(XAIRError)
async def xair_exception_handler(request: Request, exc: XAIRError):
    """Handle XAIR-specific exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=format_error_for_user(exc)
    )

@app.post("/api/generate")
async def generate_tree(request: GenerateRequest):
    """Generate a reasoning tree for a prompt"""
    try:
        # Step 1: Generate cross-generational paths
        cross_gen = await cgrt_generator.generate_cross_paths(
            request.prompt, 
            max_tokens=request.max_tokens
        )
        
        # Step 2: Build CGRT tree from generations
        tree_builder = CGRTTree(
            merge_threshold=0.8,
            importance_threshold=request.min_probability
        )
        tree = tree_builder.build_from_generations(cross_gen)
        
        # Generate tree ID
        tree_id = str(len(trees) + 1)
        
        # Store tree and cross generation data
        cross_generations[tree_id] = cross_gen
        trees[tree_id] = {
            "id": tree_id,
            "prompt": request.prompt,
            "tree": tree,
            "stats": {
                "node_count": tree.number_of_nodes(),
                "edge_count": tree.number_of_edges(),
                "shared_prefix_length": len(cross_gen.shared_prefix),
                "divergence_points": len(cross_gen.divergence_map)
            }
        }
        
        # Get most likely path (for display)
        # This is an approximation since CGRT doesn't track paths the same way
        most_likely_path = []
        if tree.nodes:
            start_node = list(tree.nodes())[0]
            # Find a path through the tree
            for node in nx.bfs_tree(tree, start_node):
                node_data = tree.nodes[node]['node']
                most_likely_path.append({
                    "id": node_data.id,
                    "text": " ".join(node_data.tokens) if hasattr(node_data, 'tokens') else "Node"
                })
        
        # Prepare response
        response = {
            "id": tree_id,
            "prompt": request.prompt,
            "stats": trees[tree_id]["stats"],
            "most_likely_path": most_likely_path,
        }
        
        # Add visualization data
        response["visualization"] = {
            "nodes": [],
            "edges": []
        }
        
        for node_id in tree.nodes():
            node_data = tree.nodes[node_id]
            if 'node' not in node_data:
                continue
                
            node = node_data['node']
            
            # Get text from tokens
            if hasattr(node, 'tokens'):
                text = " ".join(node.tokens)
            else:
                text = f"Node {node_id}"
                
            # Get probability
            if hasattr(node, 'probabilities') and node.probabilities:
                probability = sum(node.probabilities) / len(node.probabilities)
            else:
                probability = 0.5
                
            response["visualization"]["nodes"].append({
                "id": node.id,
                "text": text,
                "probability": probability,
                "importance_score": node.importance_score,
                "is_counterfactual": False  # Base tree has no counterfactuals yet
            })
            
        for source, target, data in tree.edges(data=True):
            response["visualization"]["edges"].append({
                "source": source,
                "target": target,
                "weight": data.get("weight", 1.0)
            })
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating tree: {e}", exc_info=True)
        if isinstance(e, XAIRError):
            raise
        raise HTTPException(status_code=500, detail=f"Error generating tree: {str(e)}")

@app.post("/api/counterfactuals")
async def generate_counterfactuals(request: CounterfactualRequest):
    """Generate counterfactuals for a reasoning tree"""
    try:
        # Get tree
        if request.tree_id not in trees:
            raise HTTPException(status_code=404, detail=f"Tree {request.tree_id} not found")
        
        tree_data = trees[request.tree_id]
        
        # Generate counterfactuals
        counterfactuals = await cf_generator.generate_counterfactuals(tree_data["tree"])
        
        # Store counterfactuals
        counterfactuals_store[request.tree_id] = counterfactuals
        
        # Integrate counterfactuals
        integrated_tree = await cf_integrator.integrate_counterfactuals(
            tree_data["tree"],
            counterfactuals
        )
        
        # Update stored tree
        trees[request.tree_id]["integrated_tree"] = integrated_tree
        
        # Convert counterfactuals to response format
        counterfactuals_response = []
        for cf in counterfactuals:
            counterfactuals_response.append({
                "id": cf.id,
                "original_text": cf.original_text,
                "modified_text": cf.modified_text,
                "modification_type": cf.modification_type,
                "probability": cf.probability,
                "attention_score": cf.attention_score,
                "parent_node_id": cf.parent_node_id
            })
        
        # Prepare response
        response = {
            "tree_id": request.tree_id,
            "counterfactuals": counterfactuals_response,
            "visualization": cf_integrator.get_visualization_data(integrated_tree)
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating counterfactuals: {e}", exc_info=True)
        if isinstance(e, XAIRError):
            raise
        raise HTTPException(status_code=500, detail=f"Error generating counterfactuals: {str(e)}")

@app.post("/api/analyze")
async def analyze_impacts(request: AnalyzeRequest):
    """Analyze the impact of counterfactuals"""
    try:
        # Get tree and counterfactuals
        if request.tree_id not in trees:
            raise HTTPException(status_code=404, detail=f"Tree {request.tree_id} not found")
            
        if request.tree_id not in counterfactuals_store:
            raise HTTPException(status_code=404, detail=f"Counterfactuals for tree {request.tree_id} not found")
            
        tree_data = trees[request.tree_id]
        counterfactuals = counterfactuals_store[request.tree_id]
        
        # Filter counterfactuals if IDs provided
        if request.counterfactual_ids:
            counterfactuals = [cf for cf in counterfactuals if cf.id in request.counterfactual_ids]
            
        # Get integrated tree
        integrated_tree = tree_data.get("integrated_tree", tree_data["tree"])
        
        # Analyze impacts
        impact_scores = await impact_analyzer.analyze_impacts(integrated_tree, counterfactuals)
        
        # Store impact scores
        impact_scores_store[request.tree_id] = impact_scores
        
        # Convert impact scores to response format
        impact_scores_response = []
        for score in impact_scores:
            impact_scores_response.append({
                "counterfactual_id": score.counterfactual_id,
                "local_impact": score.local_impact,
                "global_impact": score.global_impact,
                "structural_impact": score.structural_impact,
                "plausibility": score.plausibility,
                "composite_score": score.composite_score,
                "confidence": score.confidence,
                "affected_nodes": list(score.affected_nodes)
            })
            
        # Get ranked impacts
        ranked_impacts = impact_analyzer.get_ranked_impacts(impact_scores)
        
        # Prepare response
        response = {
            "tree_id": request.tree_id,
            "impact_scores": impact_scores_response,
            "ranked_impacts": [{"id": id, "score": score} for id, score in ranked_impacts],
            "explanations": [
                impact_analyzer.get_impact_explanation(score)
                for score in impact_scores
            ]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing impacts: {e}", exc_info=True)
        if isinstance(e, XAIRError):
            raise
        raise HTTPException(status_code=500, detail=f"Error analyzing impacts: {str(e)}")

# Serve frontend static files
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)