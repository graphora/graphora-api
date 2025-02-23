from functools import lru_cache
from typing import List, Type

from app.baml_client.type_builder import TypeBuilder
from pydantic import BaseModel
from app.baml_client import b
from app.baml_client.types import RelationshipInference, ResolvedEntities, StandardisedProperties
from app.utils.parse_pydantic_schema import build_from_pydantic
from app.baml_client import reset_baml_env_vars
from app.config import settings
import os
import dotenv
from aiocache import cached, Cache
import hashlib

dotenv.load_dotenv()
reset_baml_env_vars(dict(os.environ))


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

class LLMClient:
    """Client for LLM-based extraction"""
    
    @cached(ttl=86400, 
        key_builder=lambda f, *args, **kwargs: f"{md5(kwargs['chunk'])+':'+str(kwargs['response_model'])}")
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
    
    @cached(ttl=86400, 
        key_builder=lambda f, *args, **kwargs: f"{md5(kwargs['rel_type']+':'+kwargs['source_type']+':'+kwargs['source_entities']+':'+kwargs['target_type']+':'+kwargs['target_entities'])}")
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
    
    @cached(ttl=86400, 
        key_builder=lambda f, *args, **kwargs: f"{md5(kwargs['entity_group_type']+':'+kwargs['entities_json'])}")
    async def standardise_properties(
        self,
        entity_group_type: str,
        entities_json: str
    ) -> List[StandardisedProperties]:
        return b.StandardiseProperties(entity_group_type=entity_group_type,
            entities_json=entities_json)
    
    @cached(ttl=86400, 
        key_builder=lambda f, *args, **kwargs: f"{md5(kwargs['entity_type']+':'+kwargs['node_dicts_str'])}")
    async def resolve_entities(
        self,
        entity_type: str,
        node_dicts_str: str
    ) -> List[ResolvedEntities]:
        return b.ResolveEntities(entity_type=entity_type,
            node_dicts_str=node_dicts_str)