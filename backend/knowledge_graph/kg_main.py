"""
Main Knowledge Graph module for the XAIR system.
Integrates entity mapping and explanation validation.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Any, Optional, Set
import time

from backend.knowledge_graph.kg_mapper import KGMapper
from backend.knowledge_graph.validator import KnowledgeGraphValidator, ValidationResult
from backend.utils.progress_monitor import ProgressMonitor, Stage

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Main Knowledge Graph class that integrates mapping and validation.
    """

    def __init__(
        self,
        use_local_model: bool = True,
        wikidata_endpoint: str = "https://query.wikidata.org/sparql",
        min_similarity_threshold: float = 0.7,
        output_dir: str = "output",
        verbose: bool = False
    ):
        """
        Initialize the Knowledge Graph system.

        Args:
            use_local_model: Whether to use a local sentence transformer model
            wikidata_endpoint: SPARQL endpoint for Wikidata
            min_similarity_threshold: Minimum similarity threshold for validation
            output_dir: Directory to save outputs
            verbose: Whether to log detailed information
        """
        self.output_dir = output_dir
        self.verbose = verbose

        if self.verbose:
            logging.basicConfig(level=logging.INFO)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        logger.info("Initializing Knowledge Graph Mapper...")
        self.kg_mapper = KGMapper(
            use_local_model=use_local_model,
            wikidata_endpoint=wikidata_endpoint,
            cache_dir=os.path.join(output_dir, "kg_cache"),
            verbose=verbose
        )

        logger.info("Initializing Knowledge Graph Validator...")
        self.validator = KnowledgeGraphValidator(
            kg_mapper=self.kg_mapper,
            min_similarity_threshold=min_similarity_threshold,
            output_dir=output_dir,
            verbose=verbose
        )

        # State variables
        self.entity_mapping = {}  # Mapping of nodes to entities
        self.validation_results = {}  # Validation results by path

        # Create progress monitor
        self.progress_monitor = ProgressMonitor(verbose=verbose)

        logger.info("Knowledge Graph system initialized successfully")

    def process_reasoning_tree(
        self,
        tree_builder,
        paths: List[Dict[str, Any]],
        save_results: bool = True
    ):
        """
        Process a reasoning tree with the Knowledge Graph system.

        Args:
            tree_builder: CGRTBuilder instance
            paths: List of generation results
            save_results: Whether to save results to disk

        Returns:
            Tuple of (entity_mapping, validation_results)
        """
        logger.info("Processing reasoning tree with Knowledge Graph...")
        start_time = time.time()

        # 1. Map nodes to entities
        logger.info("Mapping nodes to Knowledge Graph entities...")
        self.progress_monitor.start_stage(Stage.KG_ENTITY_MAPPING)
        self.entity_mapping = self.kg_mapper.map_nodes_to_entities(tree_builder)
        self.progress_monitor.complete_stage(Stage.KG_ENTITY_MAPPING)

        # Save the entity mapping if requested
        if save_results:
            mapping_path = os.path.join(self.output_dir, "entity_mapping.json")
            self._save_entity_mapping(mapping_path)

        # 2. Validate reasoning paths
        logger.info("Validating reasoning paths...")
        self.progress_monitor.start_stage(Stage.KG_VALIDATION)
        self.validation_results = self.validator.validate_reasoning_paths(
            tree_builder,
            paths
        )
        self.progress_monitor.complete_stage(Stage.KG_VALIDATION)

        # Save validation results if requested
        if save_results:
            self.validator.save_validation_results()

            # Generate a detailed report
            report_path = os.path.join(self.output_dir, "validation_report.txt")
            self.validator.export_validation_report(report_path)

        end_time = time.time()
        logger.info(f"Knowledge Graph processing complete in {end_time - start_time:.2f}s")
        return self.entity_mapping, self.validation_results

    def get_most_trustworthy_path(self) -> Tuple[int, float]:
        """
        Get the most trustworthy reasoning path.

        Returns:
            Tuple of (path_id, trustworthiness_score)
        """
        if not self.validation_results:
            return (-1, 0.0)

        path_id = max(
            self.validation_results.keys(),
            key=lambda pid: self.validation_results[pid].trustworthiness_score
        )

        return (path_id, self.validation_results[path_id].trustworthiness_score)

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the validation results.

        Returns:
            Dictionary with validation summary
        """
        return self.validator.get_validation_summary()

    def _save_entity_mapping(self, output_path: str):
        """
        Save entity mapping to a file.

        Args:
            output_path: Path to save the mapping
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert mapping to serializable format
        serializable_mapping = {}
        for node_id, entities in self.entity_mapping.items():
            serializable_mapping[node_id] = [
                {
                    "id": entity.get("id", ""),
                    "label": entity.get("label", ""),
                    "description": entity.get("description", ""),
                    "similarity": entity.get("similarity", 0.0),
                    "uri": entity.get("uri", "")
                }
                for entity in entities
            ]

        # Save to file
        with open(output_path, "w") as f:
            json.dump(serializable_mapping, f, indent=2)

        logger.info(f"Saved entity mapping to {output_path}")

    def find_entity_relations(
        self,
        entity1_id: str,
        entity2_id: str
    ) -> List[Dict[str, Any]]:
        """
        Find relations between two entities.

        Args:
            entity1_id: ID of the first entity
            entity2_id: ID of the second entity

        Returns:
            List of relation dictionaries
        """
        return self.kg_mapper.find_relations(entity1_id, entity2_id)

    def clear_cache(self):
        """Clear the entity and relation cache."""
        self.kg_mapper.clear_cache()
        logger.info("Cleared knowledge graph cache")
