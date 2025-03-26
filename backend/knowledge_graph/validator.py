"""
Knowledge Graph Validator module for the XAIR system.
Validates reasoning paths against knowledge graph entities.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Any, Optional, Set
import numpy as np
import torch
import time
from dataclasses import dataclass, field

from backend.knowledge_graph.kg_mapper import KGMapper

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Stores validation results for a reasoning path."""
    path_id: int
    trustworthiness_score: float = 0.0
    supported_statements: List[Dict[str, Any]] = field(default_factory=list)
    contradicted_statements: List[Dict[str, Any]] = field(default_factory=list)
    unverified_statements: List[Dict[str, Any]] = field(default_factory=list)
    kg_entities_found: int = 0
    total_statements: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "path_id": self.path_id,
            "trustworthiness_score": self.trustworthiness_score,
            "supported_statements": self.supported_statements,
            "contradicted_statements": self.contradicted_statements,
            "unverified_statements": self.unverified_statements,
            "kg_entities_found": self.kg_entities_found,
            "total_statements": self.total_statements,
            "timestamp": self.timestamp
        }

class KnowledgeGraphValidator:
    """Validates reasoning paths against knowledge graph."""
    
    def __init__(
        self,
        kg_mapper: Optional[KGMapper] = None,
        min_similarity_threshold: float = 0.7,
        contradiction_threshold: float = 0.3,
        output_dir: str = "output",
        verbose: bool = False
    ):
        """
        Initialize the knowledge graph validator.
        
        Args:
            kg_mapper: KGMapper instance (will create one if not provided)
            min_similarity_threshold: Minimum similarity threshold for validation
            contradiction_threshold: Threshold for contradiction detection
            output_dir: Directory to save outputs
            verbose: Whether to log detailed information
        """
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        
        # Create KGMapper if not provided
        if kg_mapper is None:
            logger.info("Creating KGMapper instance")
            self.kg_mapper = KGMapper(verbose=verbose)
        else:
            self.kg_mapper = kg_mapper
        
        self.min_similarity_threshold = min_similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store validation results
        self.validation_results = {}
        
        # Try to import NLP libraries for statement extraction
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model for statement extraction")
        except (ImportError, OSError):
            logger.warning("Could not load spaCy model. Installing with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def validate_reasoning_paths(
        self,
        tree_builder,
        paths: List[Dict[str, Any]]
    ) -> Dict[int, ValidationResult]:
        """
        Validate multiple reasoning paths against knowledge graph.
        
        Args:
            tree_builder: CGRTBuilder instance
            paths: List of generation results
            
        Returns:
            Dictionary mapping path IDs to validation results
        """
        logger.info(f"Validating {len(paths)} reasoning paths...")
        
        # Reset validation results
        self.validation_results = {}
        
        # First, map nodes to entities
        entity_mapping = self.kg_mapper.map_nodes_to_entities(tree_builder)
        
        # Then validate each path
        for path_idx, path in enumerate(paths):
            result = self.validate_path(path_idx, path, tree_builder, entity_mapping)
            self.validation_results[path_idx] = result
            
            logger.info(f"Path {path_idx}: Trustworthiness score = {result.trustworthiness_score:.2f}")
            logger.info(f"  Supported statements: {len(result.supported_statements)}")
            logger.info(f"  Contradicted statements: {len(result.contradicted_statements)}")
            logger.info(f"  Unverified statements: {len(result.unverified_statements)}")
        
        return self.validation_results
    
    def validate_path(
        self,
        path_id: int,
        path: Dict[str, Any],
        tree_builder,
        entity_mapping: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> ValidationResult:
        """
        Validate a single reasoning path against knowledge graph.
        
        Args:
            path_id: ID of the path
            path: Path data
            tree_builder: CGRTBuilder instance
            entity_mapping: Pre-computed entity mapping (optional)
            
        Returns:
            ValidationResult object
        """
        # Extract the generated text
        text = path.get("generated_text", "")
        if not text:
            logger.warning(f"Path {path_id} has no generated text")
            return ValidationResult(path_id=path_id)
        
        # Extract statements from the text
        statements = self._extract_statements(text)
        
        # Create validation result
        result = ValidationResult(
            path_id=path_id,
            total_statements=len(statements)
        )
        
        # Find entities if not provided
        if entity_mapping is None:
            entity_mapping = self.kg_mapper.map_nodes_to_entities(tree_builder)
        
        # Get all entities from the mapping
        all_entities = []
        for node_id, entities in entity_mapping.items():
            all_entities.extend(entities)
        
        # Count unique entities
        unique_entities = {entity["id"]: entity for entity in all_entities}
        result.kg_entities_found = len(unique_entities)
        
        # Validate each statement
        for stmt in statements:
            validation = self._validate_statement(stmt, unique_entities.values())
            
            if validation["status"] == "supported":
                result.supported_statements.append({
                    "statement": stmt,
                    "entities": validation["entities"],
                    "confidence": validation["confidence"]
                })
            elif validation["status"] == "contradicted":
                result.contradicted_statements.append({
                    "statement": stmt,
                    "entities": validation["entities"],
                    "contradiction": validation["contradiction"],
                    "confidence": validation["confidence"]
                })
            else:  # unverified
                result.unverified_statements.append({
                    "statement": stmt,
                    "entities": validation["entities"] if "entities" in validation else [],
                    "reason": validation.get("reason", "No relevant knowledge found")
                })
        
        # Calculate trustworthiness score
        result.trustworthiness_score = self._calculate_trustworthiness(result)
        
        return result
    
    def _extract_statements(self, text: str) -> List[str]:
        """
        Extract statements from text for validation.
        
        Args:
            text: Text to extract statements from
            
        Returns:
            List of statement strings
        """
        statements = []
        
        # Try using spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                
                # Extract sentences
                for sent in doc.sents:
                    # Filter to likely factual statements
                    # Skip questions, imperatives, etc.
                    sent_text = sent.text.strip()
                    
                    # Skip short sentences
                    if len(sent_text) < 10:
                        continue
                    
                    # Skip sentences that don't look like statements
                    if sent_text.endswith("?") or sent_text.startswith(("If ", "Would ", "Could ", "Let ")):
                        continue
                    
                    # Add to statements
                    statements.append(sent_text)
            except Exception as e:
                logger.error(f"Error extracting statements with spaCy: {e}")
                # Fall back to simple splitting
                statements = self._simple_statement_extraction(text)
        else:
            # Simple fallback
            statements = self._simple_statement_extraction(text)
        
        return statements
    
    def _simple_statement_extraction(self, text: str) -> List[str]:
        """
        Simple statement extraction by sentence splitting.
        
        Args:
            text: Text to split into statements
            
        Returns:
            List of statement strings
        """
        # Split by common sentence terminators
        import re
        # Pattern matches sentence boundaries but keeps the delimiter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter to likely factual statements
        statements = []
        for sent in sentences:
            sent = sent.strip()
            
            # Skip short sentences
            if len(sent) < 10:
                continue
            
            # Skip sentences that don't look like statements
            if sent.endswith("?") or sent.startswith(("If ", "Would ", "Could ", "Let ")):
                continue
            
            statements.append(sent)
        
        return statements
    
    def _validate_statement(
        self,
        statement: str,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate a statement against knowledge graph entities.
        
        Args:
            statement: Statement to validate
            entities: List of entity dictionaries
            
        Returns:
            Validation result dictionary
        """
        # First, check if the statement contains any of the entities
        statement_lower = statement.lower()
        relevant_entities = []
        
        for entity in entities:
            # Check if entity label is in the statement
            entity_label = entity["label"].lower()
            if entity_label in statement_lower:
                relevant_entities.append(entity)
                continue
            
            # Check alternative labels if available
            for alt_label in entity.get("alt_labels", []):
                if alt_label.lower() in statement_lower:
                    relevant_entities.append(entity)
                    break
        
        # If no relevant entities found, mark as unverified
        if not relevant_entities:
            return {
                "status": "unverified",
                "reason": "No relevant entities found"
            }
        
        # Check for contradictions and support from entity properties
        contradictions = []
        support = []
        
        for entity in relevant_entities:
            # Skip entities without properties
            if "properties" not in entity:
                continue
            
            properties = entity.get("properties", {})
            
            # Check each property value against the statement
            for prop_name, values in properties.items():
                for value in values:
                    # Convert value to string and lowercase for comparison
                    value_str = str(value).lower()
                    
                    # Skip very short values
                    if len(value_str) < 3:
                        continue
                    
                    # Check if value is in the statement (supporting evidence)
                    if value_str in statement_lower:
                        support.append({
                            "entity": entity["label"],
                            "property": prop_name,
                            "value": value
                        })
                    
                    # Check for contradictions by looking for negations
                    # This is a simplified approach; a more sophisticated approach 
                    # would use natural language inference
                    negation_patterns = [
                        f"not {value_str}",
                        f"isn't {value_str}",
                        f"isn't a {value_str}",
                        f"isn't the {value_str}",
                        f"is not {value_str}",
                        f"is not a {value_str}",
                        f"is not the {value_str}",
                        f"doesn't {value_str}",
                        f"does not {value_str}",
                        f"didn't {value_str}",
                        f"did not {value_str}",
                        f"never {value_str}"
                    ]
                    
                    for pattern in negation_patterns:
                        if pattern in statement_lower:
                            contradictions.append({
                                "entity": entity["label"],
                                "property": prop_name,
                                "value": value,
                                "negation": pattern
                            })
        
        # Determine overall validation status
        if contradictions:
            # Contradictions take precedence
            return {
                "status": "contradicted",
                "entities": relevant_entities,
                "contradiction": contradictions[0],  # Return first contradiction
                "confidence": min(0.7, len(contradictions) * 0.2)
            }
        elif support:
            # Supporting evidence found
            return {
                "status": "supported",
                "entities": relevant_entities,
                "evidence": support,
                "confidence": min(0.9, 0.5 + len(support) * 0.1)
            }
        else:
            # Entities found but no direct support or contradiction
            return {
                "status": "unverified",
                "entities": relevant_entities,
                "reason": "No supporting evidence found"
            }
    
    def _calculate_trustworthiness(self, result: ValidationResult) -> float:
        """
        Calculate the trustworthiness score for a validation result.
        
        Args:
            result: ValidationResult object
            
        Returns:
            Trustworthiness score (0-1)
        """
        # Base score
        if result.total_statements == 0:
            return 0.5  # Neutral score if no statements
        
        # Count statements by category
        supported = len(result.supported_statements)
        contradicted = len(result.contradicted_statements)
        unverified = len(result.unverified_statements)
        
        # Calculate average confidence of supported statements
        supported_confidence = 0.0
        if supported > 0:
            supported_confidence = sum(s["confidence"] for s in result.supported_statements) / supported
        
        # Calculate average confidence of contradicted statements
        contradicted_confidence = 0.0
        if contradicted > 0:
            contradicted_confidence = sum(s["confidence"] for s in result.contradicted_statements) / contradicted
        
        # Weightings for each category
        supported_weight = 0.7
        contradicted_weight = 0.2
        unverified_weight = 0.1
        
        # Calculate verification ratio
        verification_ratio = (supported + contradicted) / result.total_statements if result.total_statements > 0 else 0
        
        # Calculate positive ratio (portion of verified statements that are supported)
        positive_ratio = supported / (supported + contradicted) if (supported + contradicted) > 0 else 0.5
        
        # Adjust positive ratio by confidence
        if supported > 0 and contradicted > 0:
            positive_ratio = (positive_ratio * supported_confidence) / (positive_ratio * supported_confidence + (1 - positive_ratio) * contradicted_confidence)
        
        # Combine factors for final score
        base_score = positive_ratio * 0.8 + 0.1  # Range from 0.1 to 0.9
        
        # Adjust based on verification ratio (more verification = more confidence in score)
        adjustment = verification_ratio * 0.2  # Max adjustment of 0.2
        
        # Apply adjustment, but ensure score stays in [0, 1] range
        score = max(0.0, min(1.0, base_score + adjustment))
        
        return score
    
    def save_validation_results(self, output_path: Optional[str] = None):
        """
        Save validation results to a file.
        
        Args:
            output_path: Path to save the results (default: output_dir/validation_results.json)
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "validation_results.json")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert results to dictionary for serialization
        serializable_results = {
            str(path_id): result.to_dict()
            for path_id, result in self.validation_results.items()
        }
        
        # Add summary
        if self.validation_results:
            avg_score = sum(r.trustworthiness_score for r in self.validation_results.values()) / len(self.validation_results)
            
            total_supported = sum(len(r.supported_statements) for r in self.validation_results.values())
            total_contradicted = sum(len(r.contradicted_statements) for r in self.validation_results.values())
            total_unverified = sum(len(r.unverified_statements) for r in self.validation_results.values())
            
            summary = {
                "average_trustworthiness": avg_score,
                "total_supported_statements": total_supported,
                "total_contradicted_statements": total_contradicted,
                "total_unverified_statements": total_unverified,
                "paths_count": len(self.validation_results),
                "timestamp": time.time()
            }
            
            serializable_results["summary"] = summary
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved validation results to {output_path}")
    
    def load_validation_results(self, input_path: str):
        """
        Load validation results from a file.
        
        Args:
            input_path: Path to load the results from
        """
        # Check if the file exists
        if not os.path.exists(input_path):
            logger.error(f"Results file {input_path} not found")
            return
        
        # Load the data
        with open(input_path, "r") as f:
            data = json.load(f)
        
        # Reset validation results
        self.validation_results = {}
        
        # Load results (skip the summary key)
        for path_id_str, result_dict in data.items():
            if path_id_str == "summary":
                continue
                
            path_id = int(path_id_str)
            
            try:
                # Create ValidationResult object
                result = ValidationResult(
                    path_id=path_id,
                    trustworthiness_score=result_dict.get("trustworthiness_score", 0.0),
                    supported_statements=result_dict.get("supported_statements", []),
                    contradicted_statements=result_dict.get("contradicted_statements", []),
                    unverified_statements=result_dict.get("unverified_statements", []),
                    kg_entities_found=result_dict.get("kg_entities_found", 0),
                    total_statements=result_dict.get("total_statements", 0),
                    timestamp=result_dict.get("timestamp", time.time())
                )
                
                self.validation_results[path_id] = result
            except Exception as e:
                logger.error(f"Error loading result for path {path_id}: {e}")
        
        logger.info(f"Loaded {len(self.validation_results)} validation results from {input_path}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of validation results.
        
        Returns:
            Dictionary with validation summary
        """
        if not self.validation_results:
            return {
                "paths_count": 0,
                "average_trustworthiness": 0.0,
                "total_supported_statements": 0,
                "total_contradicted_statements": 0,
                "total_unverified_statements": 0
            }
        
        # Calculate summary statistics
        avg_score = sum(r.trustworthiness_score for r in self.validation_results.values()) / len(self.validation_results)
        
        total_supported = sum(len(r.supported_statements) for r in self.validation_results.values())
        total_contradicted = sum(len(r.contradicted_statements) for r in self.validation_results.values())
        total_unverified = sum(len(r.unverified_statements) for r in self.validation_results.values())
        
        # Find the most trustworthy path
        most_trustworthy_path_id = max(
            self.validation_results.keys(),
            key=lambda path_id: self.validation_results[path_id].trustworthiness_score
        )
        
        most_trustworthy_score = self.validation_results[most_trustworthy_path_id].trustworthiness_score
        
        # Find the least trustworthy path
        least_trustworthy_path_id = min(
            self.validation_results.keys(),
            key=lambda path_id: self.validation_results[path_id].trustworthiness_score
        )
        
        least_trustworthy_score = self.validation_results[least_trustworthy_path_id].trustworthiness_score
        
        # Prepare summary
        summary = {
            "paths_count": len(self.validation_results),
            "average_trustworthiness": avg_score,
            "total_supported_statements": total_supported,
            "total_contradicted_statements": total_contradicted,
            "total_unverified_statements": total_unverified,
            "most_trustworthy_path": {
                "path_id": most_trustworthy_path_id,
                "score": most_trustworthy_score
            },
            "least_trustworthy_path": {
                "path_id": least_trustworthy_path_id,
                "score": least_trustworthy_score
            }
        }
        
        return summary
    
    def export_validation_report(self, output_path: Optional[str] = None) -> str:
        """
        Export a detailed validation report.
        
        Args:
            output_path: Path to save the report (default: output_dir/validation_report.txt)
            
        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "validation_report.txt")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write("Knowledge Graph Validation Report\n")
            f.write("================================\n\n")
            
            # Add summary
            summary = self.get_validation_summary()
            f.write(f"Paths analyzed: {summary['paths_count']}\n")
            f.write(f"Average trustworthiness: {summary['average_trustworthiness']:.2f}\n")
            f.write(f"Total supported statements: {summary['total_supported_statements']}\n")
            f.write(f"Total contradicted statements: {summary['total_contradicted_statements']}\n")
            f.write(f"Total unverified statements: {summary['total_unverified_statements']}\n\n")
            
            # Add details for each path
            for path_id, result in sorted(self.validation_results.items()):
                f.write(f"Path {path_id}\n")
                f.write(f"{'-' * 50}\n")
                f.write(f"Trustworthiness score: {result.trustworthiness_score:.2f}\n")
                f.write(f"KG entities found: {result.kg_entities_found}\n")
                f.write(f"Total statements: {result.total_statements}\n\n")
                
                # Supported statements
                f.write(f"Supported statements ({len(result.supported_statements)}):\n")
                for i, stmt in enumerate(result.supported_statements[:5]):  # Show top 5
                    f.write(f"{i+1}. \"{stmt['statement']}\"\n")
                    f.write(f"   Entities: {', '.join(e['label'] for e in stmt['entities'])}\n")
                    f.write(f"   Confidence: {stmt['confidence']:.2f}\n\n")
                
                if len(result.supported_statements) > 5:
                    f.write(f"   ... and {len(result.supported_statements) - 5} more\n\n")
                
                # Contradicted statements
                f.write(f"Contradicted statements ({len(result.contradicted_statements)}):\n")
                for i, stmt in enumerate(result.contradicted_statements):
                    f.write(f"{i+1}. \"{stmt['statement']}\"\n")
                    f.write(f"   Entities: {', '.join(e['label'] for e in stmt['entities'])}\n")
                    f.write(f"   Contradiction: {stmt['contradiction']['entity']} {stmt['contradiction']['property']} {stmt['contradiction']['value']}\n")
                    f.write(f"   Confidence: {stmt['confidence']:.2f}\n\n")
                
                # Add a few unverified statements
                f.write(f"Unverified statements ({len(result.unverified_statements)} total, showing 3):\n")
                for i, stmt in enumerate(result.unverified_statements[:3]):
                    f.write(f"{i+1}. \"{stmt['statement']}\"\n")
                    if stmt['entities']:
                        f.write(f"   Entities found: {', '.join(e['label'] for e in stmt['entities'])}\n")
                    f.write(f"   Reason: {stmt['reason']}\n\n")
                
                f.write("\n")
        
        logger.info(f"Exported validation report to {output_path}")
        return output_path