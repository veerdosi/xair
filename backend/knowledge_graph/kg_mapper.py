"""
Knowledge Graph Mapper module for the XAIR system.
Maps CGRT nodes to knowledge graph entities in Wikidata.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Any, Optional, Set
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from SPARQLWrapper import SPARQLWrapper, JSON
import torch
import re

logger = logging.getLogger(__name__)

class KGMapper:
    """Maps CGRT nodes to knowledge graph entities."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",  # Lightweight model for MacBook compatibility
        use_local_model: bool = True,
        wikidata_endpoint: str = "https://query.wikidata.org/sparql",
        cache_dir: str = "cache",
        cache_expiry: int = 86400,  # 24 hours in seconds
        use_gpu: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the knowledge graph mapper.
        
        Args:
            model_name: Name of the sentence transformer model
            use_local_model: Whether to use a local model or download from HuggingFace
            wikidata_endpoint: SPARQL endpoint for Wikidata
            cache_dir: Directory to cache SPARQL results
            cache_expiry: Cache expiry time in seconds
            use_gpu: Whether to use GPU for embeddings
            verbose: Whether to log detailed information
        """
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        
        # Initialize sentence transformer model for semantic similarity
        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name, device="cpu")
            # Move to GPU only if specified and available
            if use_gpu and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info("Using GPU for embeddings")
            elif use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
                logger.info("Using Apple MPS for embeddings")
            else:
                logger.info("Using CPU for embeddings")
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            logger.warning("Falling back to simple string matching for entity linking")
            self.model = None
        
        # Initialize SPARQL wrapper
        self.sparql = SPARQLWrapper(wikidata_endpoint)
        self.sparql.setReturnFormat(JSON)
        
        # Set up caching
        self.cache_dir = cache_dir
        self.cache_expiry = cache_expiry
        os.makedirs(cache_dir, exist_ok=True)
        
        # Entity and relation cache
        self.entity_cache = {}
        self.relation_cache = {}
        
        # Load existing cache from disk
        self._load_cache()
    
    def map_nodes_to_entities(
        self,
        tree_builder,
        max_nodes: int = 100,
        min_similarity: float = 0.6
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Map nodes in the reasoning tree to knowledge graph entities.
        
        Args:
            tree_builder: CGRTBuilder instance
            max_nodes: Maximum number of nodes to process
            min_similarity: Minimum similarity threshold for entity mapping
            
        Returns:
            Dictionary mapping node IDs to entity lists
        """
        logger.info("Mapping nodes to knowledge graph entities...")
        
        # Get all nodes from the tree builder
        node_mapping = {}
        processed_count = 0
        
        # Process nodes by importance (most important first)
        nodes_by_importance = sorted(
            tree_builder.nodes.items(),
            key=lambda x: x[1].importance_score,
            reverse=True
        )
        
        # Extract relevant text from nodes
        for node_id, node in nodes_by_importance[:max_nodes]:
            # Skip very short tokens
            if len(node.token.strip()) <= 1:
                continue
            
            # Get node context
            context = self._get_node_context(tree_builder, node_id)
            
            # Find entities for this node
            entities = self.find_entities(node.token, context)
            
            if entities:
                node_mapping[node_id] = entities
                
                # Also update the node in the tree builder
                node.kg_entities = entities
                
                # Update the graph node
                if node_id in tree_builder.graph:
                    tree_builder.graph.nodes[node_id]["kg_entities"] = entities
            
            processed_count += 1
            if processed_count % 10 == 0:
                logger.info(f"Processed {processed_count} nodes, found entities for {len(node_mapping)} nodes")
        
        logger.info(f"Mapped {len(node_mapping)} nodes to knowledge graph entities")
        return node_mapping
    
    def _get_node_context(
        self,
        tree_builder,
        node_id: str,
        context_size: int = 5
    ) -> str:
        """
        Get context around a node for better entity mapping.
        
        Args:
            tree_builder: CGRTBuilder instance
            node_id: ID of the node
            context_size: Number of nodes to include in context
            
        Returns:
            Context string
        """
        if node_id not in tree_builder.nodes:
            return ""
        
        node = tree_builder.nodes[node_id]
        position = node.position
        
        # Find nodes around this position
        context_nodes = []
        for other_id, other_node in tree_builder.nodes.items():
            if abs(other_node.position - position) <= context_size:
                context_nodes.append(other_node)
        
        # Sort by position
        context_nodes.sort(key=lambda n: n.position)
        
        # Combine tokens
        context = " ".join(n.token for n in context_nodes)
        return context
    
    def find_entities(
        self,
        text: str,
        context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Find knowledge graph entities for a text.
        
        Args:
            text: Text to find entities for
            context: Additional context for better matching
            
        Returns:
            List of entity dictionaries
        """
        # Clean the text
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return []
        
        # Create a combined query with context if available
        query_text = cleaned_text
        if context:
            # Extract key terms from context
            key_terms = self._extract_key_terms(context)
            if key_terms:
                query_text = f"{cleaned_text} {key_terms}"
        
        # Check cache first
        cache_key = query_text.lower()
        if cache_key in self.entity_cache:
            cached_result = self.entity_cache[cache_key]
            # Check if cache is still valid
            if time.time() - cached_result["timestamp"] < self.cache_expiry:
                return cached_result["entities"]
        
        # Search for entities in Wikidata
        entities = self._search_wikidata_entities(query_text)
        
        # Add to cache
        self.entity_cache[cache_key] = {
            "entities": entities,
            "timestamp": time.time()
        }
        
        # Save cache periodically
        if len(self.entity_cache) % 50 == 0:
            self._save_cache()
        
        return entities
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for better entity matching.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove special characters
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def _extract_key_terms(self, text: str, max_terms: int = 3) -> str:
        """
        Extract key terms from context.
        
        Args:
            text: Text to extract terms from
            max_terms: Maximum number of terms to extract
            
        Returns:
            String of key terms
        """
        # Simple extraction of longer words as potential key terms
        words = text.split()
        # Filter out stopwords and short words
        stopwords = {"the", "and", "is", "in", "to", "of", "a", "for", "with", "as", "that", "on", "at", "by", "an"}
        key_words = [w for w in words if len(w) > 4 and w.lower() not in stopwords]
        # Take the most frequent or first few
        if len(key_words) > max_terms:
            # Count word frequencies
            from collections import Counter
            counts = Counter(key_words)
            key_words = [word for word, _ in counts.most_common(max_terms)]
        return " ".join(key_words[:max_terms])
    
    def _search_wikidata_entities(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for entities in Wikidata.
        
        Args:
            query: Query string
            limit: Maximum number of results
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        try:
            # Build SPARQL query
            sparql_query = f"""
            SELECT ?item ?itemLabel ?itemDescription ?itemAltLabel WHERE {{
              SERVICE wikibase:mwapi {{
                bd:serviceParam wikibase:api "EntitySearch".
                bd:serviceParam wikibase:endpoint "www.wikidata.org".
                bd:serviceParam mwapi:search "{query}".
                bd:serviceParam mwapi:language "en".
                ?item wikibase:apiOutputItem mwapi:item.
              }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
              OPTIONAL {{ ?item skos:altLabel ?itemAltLabel . FILTER (LANG(?itemAltLabel) = "en") }}
            }}
            LIMIT {limit}
            """
            
            # Execute query
            self.sparql.setQuery(sparql_query)
            results = self.sparql.query().convert()
            
            # Process results
            for result in results["results"]["bindings"]:
                entity_id = result["item"]["value"].split("/")[-1]
                entity = {
                    "id": entity_id,
                    "uri": result["item"]["value"],
                    "label": result.get("itemLabel", {}).get("value", ""),
                    "description": result.get("itemDescription", {}).get("value", ""),
                    "alt_labels": result.get("itemAltLabel", {}).get("value", "").split(",")
                }
                
                # Skip if no label
                if not entity["label"]:
                    continue
                
                # Calculate similarity if model is available
                if self.model:
                    similarity = self._calculate_similarity(query, entity["label"])
                    entity["similarity"] = similarity
                else:
                    # Simple string matching fallback
                    query_lower = query.lower()
                    label_lower = entity["label"].lower()
                    if query_lower in label_lower or label_lower in query_lower:
                        entity["similarity"] = 0.8  # Arbitrary similarity score
                    else:
                        entity["similarity"] = 0.5
                
                entities.append(entity)
            
            # Sort by similarity
            entities.sort(key=lambda e: e["similarity"], reverse=True)
            
            # Get additional information for top entities
            top_entities = entities[:min(3, len(entities))]
            for entity in top_entities:
                self._enrich_entity(entity)
            
        except Exception as e:
            logger.error(f"Error searching Wikidata: {e}")
        
        return entities
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Encode texts to vectors
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)
            
            # Calculate cosine similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
            return cosine_similarity.item()
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _enrich_entity(self, entity: Dict[str, Any]) -> None:
        """
        Enrich entity with additional information from Wikidata.
        
        Args:
            entity: Entity dictionary to enrich
        """
        try:
            entity_id = entity["id"]
            
            # Check cache
            if entity_id in self.entity_cache:
                cached_data = self.entity_cache[entity_id]
                if "properties" in cached_data:
                    entity["properties"] = cached_data["properties"]
                    return
            
            # Get properties
            sparql_query = f"""
            SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
              wd:{entity_id} ?prop ?value .
              ?property wikibase:directClaim ?prop .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
              FILTER(STRSTARTS(STR(?property), "http://www.wikidata.org/entity/P"))
            }}
            LIMIT 20
            """
            
            self.sparql.setQuery(sparql_query)
            results = self.sparql.query().convert()
            
            properties = {}
            for result in results["results"]["bindings"]:
                prop_id = result["property"]["value"].split("/")[-1]
                prop_label = result.get("propertyLabel", {}).get("value", prop_id)
                value = result.get("valueLabel", {}).get("value", "")
                
                if prop_label not in properties:
                    properties[prop_label] = []
                
                if value and value not in properties[prop_label]:
                    properties[prop_label].append(value)
            
            entity["properties"] = properties
            
            # Update cache
            if entity_id in self.entity_cache:
                self.entity_cache[entity_id]["properties"] = properties
            else:
                self.entity_cache[entity_id] = {
                    "properties": properties,
                    "timestamp": time.time()
                }
            
        except Exception as e:
            logger.error(f"Error enriching entity: {e}")
    
    def find_relations(
        self,
        entity1_id: str,
        entity2_id: str
    ) -> List[Dict[str, Any]]:
        """
        Find relations between two entities in Wikidata.
        
        Args:
            entity1_id: ID of the first entity
            entity2_id: ID of the second entity
            
        Returns:
            List of relation dictionaries
        """
        # Create cache key
        cache_key = f"{entity1_id}_{entity2_id}"
        
        # Check cache
        if cache_key in self.relation_cache:
            cached_result = self.relation_cache[cache_key]
            if time.time() - cached_result["timestamp"] < self.cache_expiry:
                return cached_result["relations"]
        
        relations = []
        
        try:
            # Direct relations: entity1 → entity2
            sparql_query = f"""
            SELECT ?prop ?propLabel ?propDescription
            WHERE {{
              wd:{entity1_id} ?p wd:{entity2_id} .
              ?prop wikibase:directClaim ?p .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            """
            
            self.sparql.setQuery(sparql_query)
            results = self.sparql.query().convert()
            
            for result in results["results"]["bindings"]:
                prop_id = result["prop"]["value"].split("/")[-1]
                relation = {
                    "id": prop_id,
                    "uri": result["prop"]["value"],
                    "label": result.get("propLabel", {}).get("value", ""),
                    "description": result.get("propDescription", {}).get("value", ""),
                    "direction": "direct"  # entity1 → entity2
                }
                relations.append(relation)
            
            # Reverse relations: entity2 → entity1
            sparql_query = f"""
            SELECT ?prop ?propLabel ?propDescription
            WHERE {{
              wd:{entity2_id} ?p wd:{entity1_id} .
              ?prop wikibase:directClaim ?p .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            """
            
            self.sparql.setQuery(sparql_query)
            results = self.sparql.query().convert()
            
            for result in results["results"]["bindings"]:
                prop_id = result["prop"]["value"].split("/")[-1]
                relation = {
                    "id": prop_id,
                    "uri": result["prop"]["value"],
                    "label": result.get("propLabel", {}).get("value", ""),
                    "description": result.get("propDescription", {}).get("value", ""),
                    "direction": "reverse"  # entity2 → entity1
                }
                relations.append(relation)
            
            # Indirect relations (entity1 → X → entity2)
            sparql_query = f"""
            SELECT ?intermediate ?intermediateLabel ?prop1 ?prop1Label ?prop2 ?prop2Label
            WHERE {{
              wd:{entity1_id} ?p1 ?intermediate .
              ?intermediate ?p2 wd:{entity2_id} .
              
              ?prop1 wikibase:directClaim ?p1 .
              ?prop2 wikibase:directClaim ?p2 .
              
              FILTER(?intermediate != wd:{entity1_id} && ?intermediate != wd:{entity2_id})
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 5
            """
            
            self.sparql.setQuery(sparql_query)
            results = self.sparql.query().convert()
            
            for result in results["results"]["bindings"]:
                intermediate_id = result["intermediate"]["value"].split("/")[-1]
                prop1_id = result["prop1"]["value"].split("/")[-1]
                prop2_id = result["prop2"]["value"].split("/")[-1]
                
                relation = {
                    "type": "indirect",
                    "intermediate": {
                        "id": intermediate_id,
                        "label": result.get("intermediateLabel", {}).get("value", "")
                    },
                    "first_relation": {
                        "id": prop1_id,
                        "label": result.get("prop1Label", {}).get("value", "")
                    },
                    "second_relation": {
                        "id": prop2_id,
                        "label": result.get("prop2Label", {}).get("value", "")
                    }
                }
                relations.append(relation)
            
        except Exception as e:
            logger.error(f"Error finding relations: {e}")
        
        # Cache results
        self.relation_cache[cache_key] = {
            "relations": relations,
            "timestamp": time.time()
        }
        
        return relations
    
    def _load_cache(self):
        """Load entity and relation cache from disk."""
        entity_cache_path = os.path.join(self.cache_dir, "entity_cache.json")
        relation_cache_path = os.path.join(self.cache_dir, "relation_cache.json")
        
        if os.path.exists(entity_cache_path):
            try:
                with open(entity_cache_path, "r") as f:
                    self.entity_cache = json.load(f)
                logger.info(f"Loaded {len(self.entity_cache)} entities from cache")
            except Exception as e:
                logger.error(f"Error loading entity cache: {e}")
                self.entity_cache = {}
        
        if os.path.exists(relation_cache_path):
            try:
                with open(relation_cache_path, "r") as f:
                    self.relation_cache = json.load(f)
                logger.info(f"Loaded {len(self.relation_cache)} relations from cache")
            except Exception as e:
                logger.error(f"Error loading relation cache: {e}")
                self.relation_cache = {}
    
    def _save_cache(self):
        """Save entity and relation cache to disk."""
        entity_cache_path = os.path.join(self.cache_dir, "entity_cache.json")
        relation_cache_path = os.path.join(self.cache_dir, "relation_cache.json")
        
        try:
            with open(entity_cache_path, "w") as f:
                json.dump(self.entity_cache, f)
            
            with open(relation_cache_path, "w") as f:
                json.dump(self.relation_cache, f)
            
            logger.info("Saved cache to disk")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def clear_cache(self):
        """Clear the entity and relation cache."""
        self.entity_cache = {}
        self.relation_cache = {}
        
        # Remove cache files
        entity_cache_path = os.path.join(self.cache_dir, "entity_cache.json")
        relation_cache_path = os.path.join(self.cache_dir, "relation_cache.json")
        
        if os.path.exists(entity_cache_path):
            os.remove(entity_cache_path)
        
        if os.path.exists(relation_cache_path):
            os.remove(relation_cache_path)
        
        logger.info("Cleared cache")