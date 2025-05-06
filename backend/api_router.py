"""
API Router for the XAI Chat Interface.

This module provides a simple API for the frontend to interact with the XAI system.
It uses Flask to serve the API endpoints and integrates with the CGRT, Counterfactual,
and Knowledge Graph components.
"""

import os
import json
import logging
import uuid
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import XAI components
from backend.models.llm_interface import LlamaInterface, GenerationConfig
from backend.cgrt.cgrt_main import CGRT
from backend.counterfactual.counterfactual_main import Counterfactual
from backend.knowledge_graph.kg_main import KnowledgeGraph

# Import utilities
from backend.utils.config import XAIRConfig
from backend.utils.logging_utils import setup_logger
from backend.utils.error_utils import handle_exceptions, XAIRError

# Setup logger
logger = setup_logger(name="xai_api", level=logging.INFO, use_rich=True)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Initialize global components
llm = None
cgrt = None
counterfactual = None
knowledge_graph = None

# In-memory storage for message history and visualization data
message_history = {}
visualization_data = {}

@app.route('/')
def index():
    """Serve the frontend application."""
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from the frontend directory."""
    return send_from_directory('../frontend', path)

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Process a chat message and return the response with visualization data.

    Request body:
    {
        "message": "User message",
        "settings": {
            "model": {...},
            "cgrt": {...},
            "counterfactual": {...},
            "knowledgeGraph": {...}
        }
    }

    Response:
    {
        "id": "message_id",
        "text": "AI response",
        "visualization": {
            "type": "reasoning-tree",
            "data": {...}
        }
    }
    """
    global llm, cgrt, counterfactual, knowledge_graph

    try:
        # Parse request
        data = request.json
        message = data.get('message')
        settings = data.get('settings')

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Apply settings if provided
        if settings:
            apply_settings(settings)

        # Ensure components are initialized
        if not llm or not cgrt or not counterfactual:
            initialize_components()

        # Generate message ID
        message_id = f"msg-{uuid.uuid4()}"

        # Process message with CGRT
        logger.info(f"Processing message: {message[:50]}...")
        tree = cgrt.process_input(message)

        # Get reasoning paths
        paths = cgrt.get_paths_text()
        if not paths:
            return jsonify({'error': 'Failed to generate reasoning paths'}), 500

        # Use first path as response text
        response_text = paths[0]

        # Process with counterfactual
        counterfactuals = counterfactual.generate_counterfactuals(
            cgrt.tree_builder,
            llm,
            message,
            cgrt.paths
        )

        # Get top counterfactuals
        top_cfs = counterfactual.get_top_counterfactuals(5)

        # Process with knowledge graph if available
        validation_results = None
        if knowledge_graph:
            try:
                entity_mapping, validation_results = knowledge_graph.process_reasoning_tree(
                    cgrt.tree_builder,
                    cgrt.paths
                )
            except Exception as e:
                logger.error(f"Error in Knowledge Graph processing: {e}")

        # Prepare tree visualization data
        viz_data = cgrt.to_dependentree_format()

        # Store visualization data for future access
        visualization_data[message_id] = {
            'reasoning_tree': viz_data,
            'paths': paths,
            'counterfactuals': [cf.__dict__ for cf in top_cfs] if top_cfs else [],
            'validation_results': validation_results
        }

        # Store in message history
        message_history[message_id] = {
            'user_message': message,
            'response_text': response_text,
            'timestamp': None  # Set this if needed
        }

        # Prepare response
        response = {
            'id': message_id,
            'text': response_text,
            'visualization': {
                'type': 'reasoning-tree',
                'data': viz_data
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization', methods=['POST'])
def visualization():
    """
    Get visualization data for a specific message and visualization type.

    Request body:
    {
        "messageId": "message_id",
        "vizType": "reasoning-tree|token-importance|counterfactual|knowledge-graph|divergence-points",
        "options": {
            "importanceThreshold": 0.5,
            ...
        }
    }

    Response:
    {
        "type": "visualization_type",
        "data": {...}
    }
    """
    try:
        # Parse request
        data = request.json
        message_id = data.get('messageId')
        viz_type = data.get('vizType', 'reasoning-tree')
        options = data.get('options', {})

        if not message_id:
            return jsonify({'error': 'No message ID provided'}), 400

        # Check if visualization data exists for this message
        if message_id not in visualization_data:
            return jsonify({'error': 'No visualization data found for this message'}), 404

        message_viz_data = visualization_data[message_id]

        # Get data based on visualization type
        if viz_type == 'reasoning-tree':
            response_data = message_viz_data.get('reasoning_tree')
        elif viz_type == 'token-importance':
            # Extract token importance data
            tree_data = message_viz_data.get('reasoning_tree', {})
            nodes = tree_data.get('nodes', [])

            # Format data for token importance visualization
            response_data = {
                'tokens': [node.get('text', node.get('token', '')) for node in nodes],
                'importance': [node.get('importance', 0) for node in nodes],
                'attention': [node.get('attention_score', 0) for node in nodes]
            }
        elif viz_type == 'counterfactual':
            response_data = message_viz_data.get('counterfactuals', [])
        elif viz_type == 'knowledge-graph':
            response_data = message_viz_data.get('validation_results')
        elif viz_type == 'divergence-points':
            # Extract divergence points from tree data
            tree_data = message_viz_data.get('reasoning_tree', {})
            nodes = tree_data.get('nodes', [])

            # Format data for divergence points visualization
            divergence_nodes = [node for node in nodes if node.get('is_divergence_point')]
            response_data = {
                'points': divergence_nodes,
                'original_tokens': [node.get('text', node.get('token', '')) for node in nodes]
            }
        else:
            return jsonify({'error': f'Unsupported visualization type: {viz_type}'}), 400

        if not response_data:
            return jsonify({'error': f'No data available for visualization type: {viz_type}'}), 404

        # Prepare response
        response = {
            'type': viz_type,
            'data': response_data
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/counterfactual', methods=['POST'])
def get_counterfactuals():
    """
    Get counterfactual analysis for a specific message.

    Request body:
    {
        "messageId": "message_id",
        "options": {
            "topKTokens": 5,
            "attentionThreshold": 0.3,
            "maxCounterfactuals": 20
        }
    }

    Response:
    {
        "counterfactuals": [...],
        "metrics": {
            "cfr": 0.25,
            ...
        }
    }
    """
    try:
        # Parse request
        data = request.json
        message_id = data.get('messageId')
        options = data.get('options', {})

        if not message_id:
            return jsonify({'error': 'No message ID provided'}), 400

        # Check if visualization data exists for this message
        if message_id not in visualization_data:
            return jsonify({'error': 'No visualization data found for this message'}), 404

        # Get counterfactuals from stored data
        counterfactuals = visualization_data[message_id].get('counterfactuals', [])

        if not counterfactuals:
            return jsonify({'error': 'No counterfactual data available for this message'}), 404

        # Calculate CFR if available
        cfr = 0.0
        flipped_count = sum(1 for cf in counterfactuals if cf.get('flipped_output'))
        if counterfactuals:
            cfr = flipped_count / len(counterfactuals)

        # Prepare response
        response = {
            'counterfactuals': counterfactuals,
            'metrics': {
                'cfr': cfr,
                'total': len(counterfactuals),
                'flipped': flipped_count
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error getting counterfactual analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge-graph', methods=['POST'])
def get_knowledge_graph():
    """
    Get knowledge graph validation for a specific message.

    Request body:
    {
        "messageId": "message_id",
        "options": {
            "useLocalModel": true,
            "similarityThreshold": 0.6
        }
    }

    Response:
    {
        "validationResults": {...},
        "summary": {
            "averageTrustworthiness": 0.75,
            ...
        }
    }
    """
    try:
        # Parse request
        data = request.json
        message_id = data.get('messageId')
        options = data.get('options', {})

        if not message_id:
            return jsonify({'error': 'No message ID provided'}), 400

        # Check if visualization data exists for this message
        if message_id not in visualization_data:
            return jsonify({'error': 'No visualization data found for this message'}), 404

        # Get validation results from stored data
        validation_results = visualization_data[message_id].get('validation_results')

        if not validation_results:
            return jsonify({'error': 'No knowledge graph validation data available for this message'}), 404

        # Get validation summary if knowledge graph component is available
        summary = {}
        if knowledge_graph:
            try:
                summary = knowledge_graph.get_validation_summary()
            except Exception as e:
                logger.error(f"Error getting validation summary: {e}")

        # Prepare response
        response = {
            'validationResults': validation_results,
            'summary': summary
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error getting knowledge graph validation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """
    Update system settings.

    Request body:
    {
        "model": {...},
        "cgrt": {...},
        "counterfactual": {...},
        "knowledgeGraph": {...}
    }

    Response:
    {
        "success": true,
        "message": "Settings updated successfully"
    }
    """
    try:
        # Parse request
        settings = request.json

        if not settings:
            return jsonify({'error': 'No settings provided'}), 400

        # Apply settings
        apply_settings(settings)

        # Reinitialize components if needed
        initialize_components()

        return jsonify({
            'success': True,
            'message': 'Settings updated successfully'
        })

    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({'error': str(e)}), 500

def apply_settings(settings: Dict[str, Any]):
    """
    Apply settings to global config.

    Args:
        settings: Settings dictionary
    """
    config = get_config()

    # Update model settings
    if 'model' in settings:
        model_settings = settings['model']
        if 'name' in model_settings:
            config.model_name_or_path = model_settings['name']
        if 'maxTokens' in model_settings:
            config.max_tokens = model_settings['maxTokens']
        if 'temperatures' in model_settings:
            config.cgrt.temperatures = model_settings['temperatures']
        if 'pathsPerTemp' in model_settings:
            config.cgrt.paths_per_temp = model_settings['pathsPerTemp']

    # Update CGRT settings
    if 'cgrt' in settings:
        cgrt_settings = settings['cgrt']
        for key, value in cgrt_settings.items():
            if hasattr(config.cgrt, key):
                setattr(config.cgrt, key, value)

    # Update counterfactual settings
    if 'counterfactual' in settings:
        cf_settings = settings['counterfactual']
        if 'topKTokens' in cf_settings:
            config.counterfactual.top_k_tokens = cf_settings['topKTokens']
        if 'attentionThreshold' in cf_settings:
            config.counterfactual.min_attention_threshold = cf_settings['attentionThreshold']
        if 'maxCounterfactuals' in cf_settings:
            config.counterfactual.max_total_candidates = cf_settings['maxCounterfactuals']

    # Update knowledge graph settings
    if 'knowledgeGraph' in settings:
        kg_settings = settings['knowledgeGraph']
        if 'useLocalModel' in kg_settings:
            config.knowledge_graph.use_local_model = kg_settings['useLocalModel']
        if 'similarityThreshold' in kg_settings:
            config.knowledge_graph.min_similarity_threshold = kg_settings['similarityThreshold']
        if 'skip' in kg_settings:
            config.skip_kg = kg_settings['skip']

def get_config() -> XAIRConfig:
    """
    Get global config instance.

    Returns:
        Global config instance
    """
    # Check if config exists in app context
    if not hasattr(app.config, 'xair_config'):
        # Create default config
        app.config.xair_config = XAIRConfig()

        # Set output directory
        app.config.xair_config.output_dir = os.path.join('output', 'api')

    return app.config.xair_config

def initialize_components():
    """Initialize XAI components."""
    global llm, cgrt, counterfactual, knowledge_graph

    config = get_config()

    # Initialize LLM interface
    if llm is None:
        logger.info("Initializing LLM interface...")
        llm = LlamaInterface(
            model_name_or_path=config.model_name_or_path,
            device=config.device,
            load_in_4bit=False,  # Explicitly set to False for Mac compatibility
            verbose=config.verbose
        )

    # Initialize CGRT
    if cgrt is None:
        logger.info("Initializing CGRT...")
        cgrt = CGRT(
            model_name_or_path=config.model_name_or_path,
            device=config.device,
            temperatures=config.cgrt.temperatures,
            paths_per_temp=config.cgrt.paths_per_temp,
            max_new_tokens=config.max_tokens,
            output_dir=os.path.join(config.output_dir, 'cgrt'),
            verbose=config.verbose
        )

    # Initialize Counterfactual
    if counterfactual is None:
        logger.info("Initializing Counterfactual...")
        counterfactual = Counterfactual(
            top_k_tokens=config.counterfactual.top_k_tokens,
            min_attention_threshold=config.counterfactual.min_attention_threshold,
            max_total_candidates=config.counterfactual.max_total_candidates,
            output_dir=os.path.join(config.output_dir, 'counterfactual'),
            verbose=config.verbose
        )

    # Initialize Knowledge Graph if not skipped
    if knowledge_graph is None and not config.skip_kg:
        logger.info("Initializing Knowledge Graph...")
        try:
            knowledge_graph = KnowledgeGraph(
                use_local_model=config.knowledge_graph.use_local_model,
                min_similarity_threshold=config.knowledge_graph.min_similarity_threshold,
                output_dir=os.path.join(config.output_dir, 'knowledge_graph'),
                verbose=config.verbose
            )
        except Exception as e:
            logger.error(f"Error initializing Knowledge Graph: {e}")
            logger.warning("Knowledge Graph will be disabled.")

if __name__ == '__main__':
    # Create output directory
    config = get_config()
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize components
    initialize_components()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
