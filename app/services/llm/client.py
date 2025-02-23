from typing import Any, List, Type, Optional
from app.baml_client.type_builder import TypeBuilder
from pydantic import BaseModel
from app.baml_client import b
from app.baml_client.types import RelationshipInference, ResolvedEntities, StandardisedProperties
from app.utils.parse_pydantic_schema import build_from_pydantic
from app.utils.llm_client_service import generate_text
from app.baml_client import reset_baml_env_vars
import os
import dotenv

dotenv.load_dotenv()
reset_baml_env_vars(dict(os.environ))

class LLMClient:
    """Client for LLM-based extraction"""
    async def extract_from_chunk(
        self,
        chunk: str,
        response_model: Type[BaseModel],
        context: str = ""
    ) -> BaseModel:
        """Extract entities and relationships from text chunk"""
        tb = TypeBuilder()
        res = build_from_pydantic(response_model, tb)
        tb.DynamicContainer.add_property("data", res)
        result = b.ExtractChunk(chunk, context, {"tb": tb})
        return response_model.model_validate(result.data)
    
    async def infer_relationship(
        self,
        rel_type: str,
        source_type: str = "",
        source_entities: str = "",
        target_type: str = "",
        target_entities: str = "",
        existing_rels: str = ""
    ) -> List[RelationshipInference]:
        return b.InferRelationship(rel_type=rel_type,
            source_type=source_type,
            source_entities=source_entities,
            target_type=target_type,
            target_entities=target_entities,
            existing_rels=existing_rels)
        
    async def standardise_properties(
        self,
        entity_group_type: str,
        entities_json: str
    ) -> List[StandardisedProperties]:
        return b.StandardiseProperties(entity_group_type=entity_group_type,
            entities_json=entities_json)
        
    async def resolve_entities(
        self,
        entity_type: str,
        node_dicts_str: str
    ) -> List[ResolvedEntities]:
        return b.ResolveEntities(entity_type=entity_type,
            node_dicts_str=node_dicts_str)