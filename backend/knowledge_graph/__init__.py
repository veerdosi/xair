"""
Knowledge Graph package for the XAIR system.
Provides entity mapping and reasoning verification capabilities.
"""

from backend.knowledge_graph.kg_mapper import KGMapper
from backend.knowledge_graph.validator import KnowledgeGraphValidator, ValidationResult
from backend.knowledge_graph.kg_main import KnowledgeGraph

__all__ = [
    'KGMapper',
    'KnowledgeGraphValidator',
    'ValidationResult',
    'KnowledgeGraph'
]