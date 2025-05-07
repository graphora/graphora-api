from typing import List, Type, Optional

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
import pathlib
from google.genai import types
from google import genai
from app.utils.func_helper import retry_async
dotenv.load_dotenv()
reset_baml_env_vars(dict(os.environ))


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

class LLMClient:
    """Client for LLM-based extraction"""

    @cached(ttl=86400, 
        key_builder=lambda f, *args, **kwargs: f"{md5(args[1])+':'+str(kwargs['response_model'])}")
    @retry_async(max_attempts=5, delay=2, backoff=2, exceptions=(ValueError, Exception))  # Adjust exceptions
    async def extract_nodes_from_pdf(
        self,
        pdf_path: str,
        response_model: Type[BaseModel],
        ontology_yaml: str,
        context: str = "",
        model_id: str = settings.VERTEXAI_DEFAULT_MODEL,
    ) -> BaseModel:
        """Extract entities and relationships from PDF"""
        client = genai.Client(
            vertexai=True, 
            project=settings.VERTEXAI_PROJECT_ID, 
            location=settings.VERTEXAI_LOCATION,
        )
        filepath = pathlib.Path(pdf_path)
        file = types.Part.from_bytes(
            data=filepath.read_bytes(),
            mime_type='application/pdf',
        )
        # Generate a structured response using the Gemini API
        prompt = f"""
        Extract structured information from the PDF file according to the ontology specification.
        <ontology>
        {ontology_yaml}
        </ontology>

        Format the output as a JSON object with the following structure:
        1. For each entity type, include a list field named "<entity_type>_list" containing all instances
        2. Include metadata fields:
            - extraction_timestamp: Current timestamp in ISOformat string
            - tokens_used: Number of tokens used (if available)
            - confidence_score: Overall confidence in extraction (0.0 to 1.0)
        3. Omit optional fields if information is not clearly present
        4. No additional properties. Just the specified fields.

        When extracting new information, maintain consistency with these previously identified entities. If previous Entities miss any properties add them. 
        These Nodes were identified from the previous text chunks of the same doc. Use the `id` field to refer & match the nodes below.
        ```
        {context}
        ```
        
        <rules>
        - For name fields, sometimes names may be written as `LASTNAME, FIRSTNAME`. So interpret accordingly
        - Only extract information that is explicitly present in the text
        - Set confidence scores based on certainty of extraction
        - Include all required fields for each node
        - Omit optional fields if information is not clearly present
        </rules>
        """
        response = client.models.generate_content(model=model_id, 
                                                  contents=[file, prompt], 
                                                  config={'response_mime_type': 'application/json', 'response_schema': response_model})
        # Convert the response to the pydantic model and return it
        print('*'*30)
        print(response)
        print('*'*30)
        print(response.usage_metadata)
        print('*'*30)
        output_perc = 3
        if response.usage_metadata:
            output_perc = (response.usage_metadata.candidates_token_count * 100) / response.usage_metadata.total_token_count
        if response.parsed is None or output_perc < 4:
            raise ValueError("Incorrect response parsed")
        return response.parsed
    

    @cached(ttl=86400, 
        key_builder=lambda f, *args, **kwargs: f"{md5(args[1])+':'+str(kwargs['response_model'])}")
    @retry_async(max_attempts=5, delay=2, backoff=2, exceptions=(ValueError, Exception))  # Adjust exceptions
    async def extract_relationships_from_pdf(
        self,
        pdf_path: str,
        response_model: Type[BaseModel],
        ontology_yaml: str,
        context: str = "",
        model_id: str = settings.VERTEXAI_DEFAULT_MODEL,
    ) -> BaseModel:
        """Extract entities and relationships from PDF"""
        client = genai.Client(
            vertexai=True, 
            project=settings.VERTEXAI_PROJECT_ID, 
            location=settings.VERTEXAI_LOCATION,
        )
        filepath = pathlib.Path(pdf_path)
        file = types.Part.from_bytes(
            data=filepath.read_bytes(),
            mime_type='application/pdf',
        )
        # Generate a structured response using the Gemini API
        prompt = f"""
        Extract structured information from the PDF file according to the ontology specification.
        <ontology>
        {ontology_yaml}
        </ontology>

        These nodes & relationships were identified from the previous text chunks of the same doc.
        Format: (:source_node_type {{node_id & properties}}) - [:relationship_type {{properties}}] -> (:target_node_type {{node_id & properties}})
        ```
        {context}
        ```

        Output a JSON object following these rules:
        1. "<source>_<relationship>_<target>" for each relationship (e.g., "Company_HAS_BUSINESS_Business").
        2. Each relationship MUST include:
        - "source_id": The ID of the source node from the provided entities.
        - "target_id": The ID of the target node from the provided entities.
        - "properties": Any additional relationship properties (optional).
        3. Metadata: "extraction_timestamp" (ISO), "tokens_used", "confidence_score" (0.0-1.0).

        Remember:
        - Only extract information that is explicitly present in the text
        - Set confidence scores based on certainty of extraction
        - Include all required fields for each relationship
        - Omit optional fields if information is not clearly present
        """
        response = client.models.generate_content(model=model_id, 
                                                  contents=[file, prompt], 
                                                  config={'response_mime_type': 'application/json', 'response_schema': response_model})
        # Convert the response to the pydantic model and return it
        print('*'*30)
        print(response)
        print('*'*30)
        print(response.usage_metadata)
        print('*'*30)
        output_perc = 3
        if response.usage_metadata:
            output_perc = (response.usage_metadata.candidates_token_count * 100) / response.usage_metadata.total_token_count
        if response.parsed is None or output_perc < 4:
            raise ValueError("Incorrect response parsed")
        return response.parsed
    
    @cached(ttl=86400, 
        key_builder=lambda f, *args, **kwargs: f"{md5(args[1])+':'+str(kwargs['response_model'])}")
    async def extract_nodes_from_chunk(
        self,
        chunk: str,
        response_model: Type[BaseModel],
        ontology_yaml: Optional[str] = None,
        context: str = ""
    ) -> BaseModel:
        """Extract entities from text chunk"""
        tb = TypeBuilder()
        res = build_from_pydantic(response_model, tb)
        tb.DynamicContainer.add_property("data", res)
        result = b.ExtractNodesFromChunk(chunk, context, {"tb": tb})
        return response_model.model_validate(result.data)
    
    @cached(ttl=86400, 
        key_builder=lambda f, *args, **kwargs: f"{md5(args[1])+':'+str(kwargs['response_model'])}")
    async def extract_relationships_from_chunk(
        self,
        chunk: str,
        response_model: Type[BaseModel],
        ontology_yaml: Optional[str] = None,
        context: str = ""
    ) -> BaseModel:
        """Extract relationships from text chunk"""
        tb = TypeBuilder()
        res = build_from_pydantic(response_model, tb)
        tb.DynamicContainer.add_property("data", res)
        result = b.ExtractRelationshipsFromChunk(chunk, context, {"tb": tb})
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