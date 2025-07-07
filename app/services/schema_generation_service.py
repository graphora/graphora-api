import logging
import time
import uuid
import yaml
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.schemas.schema import (
    SchemaGenerationRequest, 
    SchemaGenerationResponse, 
    RelatedSchema,
    UserResponse,
    SchemaGenerationContext
)
from app.services.schema_storage_service import schema_storage_service
from app.services.schema_search_service import schema_search_service
from app.services.question_sets import QUESTION_SETS
from app.utils.llm_helper import get_user_llm_credentials, create_gemini_client
from app.utils.llm_usage_tracker import track_gemini_usage
from app.utils.baml_helper import create_baml_client, configure_baml_client_for_refinement
from app.config import settings

logger = logging.getLogger(__name__)


class SchemaGenerationService:
    """Service for AI-powered schema generation"""
    
    def __init__(self):
        self.default_templates = self._load_default_templates()
    
    async def generate_schema(
        self, 
        user_id: str, 
        request: SchemaGenerationRequest
    ) -> SchemaGenerationResponse:
        """Generate a schema based on user responses and context"""
        
        try:
            start_time = time.time()
            schema_id = str(uuid.uuid4())
            
            # Extract context from user responses
            context = self._extract_context_from_responses(request.user_responses)
            if request.context:
                context.update(request.context.model_dump(exclude_unset=True))
            
            # Search for related schemas
            related_schemas = await self._find_related_schemas(user_id, context)
            
            # Generate schema using LLM
            generated_schema, confidence, suggestions = await self._generate_schema_with_llm(
                user_id=user_id,
                context=context,
                related_schemas=related_schemas,
                user_responses=request.user_responses
            )
            
            # Store the generated schema for future reference
            await schema_storage_service.store_generated_schema(
                schema_id=schema_id,
                user_id=user_id,
                schema_content=generated_schema,
                context=context,
                confidence=confidence
            )
            
            # Prepare response
            response = SchemaGenerationResponse(
                id=schema_id,
                schema_content=generated_schema,
                confidence=confidence,
                related_schemas=related_schemas,
                suggestions=suggestions,
                metadata={
                    "generation_time_ms": int((time.time() - start_time) * 1000),
                    "context": context,
                    "total_related_schemas": len(related_schemas)
                }
            )
            
            logger.info(
                f"Generated schema for user {user_id}: {schema_id} "
                f"(confidence: {confidence:.2f}, time: {response.metadata['generation_time_ms']}ms)"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating schema for user {user_id}: {str(e)}")
            raise
    
    async def refine_schema(
        self,
        user_id: str,
        schema_id: str,
        current_schema: str,
        user_feedback: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[str], float, str]:
        """Refine an existing schema based on user feedback"""
        
        try:
            # Get user's LLM credentials and setup BAML client for refinement
            api_key, model_name = await get_user_llm_credentials(user_id)
            client_registry = configure_baml_client_for_refinement(api_key, model_name)
            
            # Extract context for refinement
            context_dict = context or {}
            use_case = context_dict.get('use_case', 'Not specified')
            domain = context_dict.get('domain', 'Not specified')
            
            # Import BAML client locally to avoid module-level import issues
            from app.baml_client import b
            
            # Call BAML structured refinement
            result = b.RefineKnowledgeGraphSchema(
                current_schema_yaml=current_schema,
                user_feedback=user_feedback,
                use_case=str(use_case),
                domain=str(domain),
                baml_options={"client_registry": client_registry}
            )
            
            # Convert structured result to YAML
            refined_schema = self._convert_structured_schema_to_yaml(result.generated_schema)
            
            # Track LLM usage (simplified since BAML handles the LLM call)
            await track_gemini_usage(
                user_id=user_id,
                model_name=model_name,
                operation_type="schema_refinement",
                response=None,  # BAML handles this internally
                operation_context="schema_refinement"
            )
            
            # Extract refinement details
            changes_made = [result.explanation] if result.explanation else []
            confidence = result.confidence
            explanation = result.explanation or "Schema refined successfully"
            
            # Update stored schema
            await schema_storage_service.update_generated_schema(
                schema_id=schema_id,
                user_id=user_id,
                updated_content=refined_schema,
                refinement_metadata={
                    "user_feedback": user_feedback,
                    "changes_made": changes_made,
                    "confidence": confidence
                }
            )
            
            return refined_schema, changes_made, confidence, explanation
            
        except Exception as e:
            logger.error(f"Error refining schema {schema_id} for user {user_id}: {str(e)}")
            raise
    
    def _extract_context_from_responses(self, responses: List[UserResponse]) -> Dict[str, Any]:
        """Extract context information from user responses"""
        context = {}
        
        for response in responses:
            question_id = response.question_id
            value = response.value
            
            # Map question IDs to context keys
            context_mapping = {
                'use_case': 'use_case',
                'data_sources': 'data_sources', 
                'domain': 'domain',
                'key_entities': 'key_entities',
                'relationships': 'relationships',
                'query_patterns': 'query_patterns',
                'data_volume': 'data_volume',
                'data_complexity': 'data_complexity',
                'temporal_requirements': 'temporal_requirements'
            }
            
            if question_id in context_mapping:
                context[context_mapping[question_id]] = value
        
        return context
    
    async def _find_related_schemas(
        self, 
        user_id: str, 
        context: Dict[str, Any]
    ) -> List[RelatedSchema]:
        """Find schemas related to the user's context"""
        
        try:
            # Build search query from context
            query_parts = []
            
            if 'domain' in context:
                query_parts.append(str(context['domain']))
            if 'use_case' in context:
                query_parts.append(str(context['use_case']))
            if 'key_entities' in context:
                entities = str(context['key_entities']).split(',')[:5]  # Limit to first 5
                query_parts.extend([entity.strip() for entity in entities])
            
            search_query = ' '.join(query_parts)
            
            if not search_query.strip():
                return []
            
            # Search for related schemas
            search_results = await schema_search_service.search_schemas(
                user_id=user_id,
                query=search_query,
                domain=context.get('domain'),
                limit=5,
                threshold=0.6
            )
            
            # Convert search results to RelatedSchema format
            related_schemas = []
            for result in search_results.results:
                related_schemas.append(RelatedSchema(
                    id=result.id,
                    title=result.title,
                    description=result.description,
                    similarity=result.similarity,
                    domain=result.domain,
                    tags=result.tags,
                    usage_count=result.usage_count
                ))
            
            return related_schemas
            
        except Exception as e:
            logger.warning(f"Error finding related schemas: {str(e)}")
            return []
    
    async def _generate_schema_with_llm(
        self,
        user_id: str,
        context: Dict[str, Any],
        related_schemas: List[RelatedSchema],
        user_responses: List[UserResponse]
    ) -> Tuple[str, float, List[str]]:
        """Generate schema using BAML structured output for deterministic YAML format"""
        
        try:
            # Get user's LLM credentials and setup BAML client
            api_key, model_name = await get_user_llm_credentials(user_id)
            client_registry = create_baml_client(api_key, model_name)
            
            # Prepare related schemas string
            related_schemas_text = ""
            for schema in related_schemas[:3]:  # Include top 3 related schemas
                related_schemas_text += f"\n- {schema.title}: {schema.description} (similarity: {schema.similarity:.2f})"
            
            # Import BAML client locally to avoid module-level import issues
            from app.baml_client import b
            
            # Call BAML structured generation with client registry
            result = b.GenerateKnowledgeGraphSchema(
                use_case=str(context.get('use_case', 'Not specified')),
                domain=str(context.get('domain', 'Not specified')), 
                data_sources=str(context.get('data_sources', 'Not specified')),
                key_entities=str(context.get('key_entities', 'Not specified')),
                relationships=str(context.get('relationships', 'Not specified')),
                query_patterns=str(context.get('query_patterns', 'Not specified')),
                data_complexity=str(context.get('data_complexity', 'Not specified')),
                data_volume=str(context.get('data_volume', 'Not specified')),
                temporal_requirements=str(context.get('temporal_requirements', 'Not specified')),
                related_schemas=related_schemas_text,
                baml_options={"client_registry": client_registry}
            )
            
            # Convert structured result to YAML
            generated_schema = self._convert_structured_schema_to_yaml(result.generated_schema)
            
            # Track usage (simplified since BAML handles the LLM call)
            await track_gemini_usage(
                user_id=user_id,
                model_name=model_name,
                operation_type="schema_generation",
                response=None,  # BAML handles this internally
                operation_context="schema_generation"
            )
            
            return generated_schema, result.confidence, result.suggestions
            
        except Exception as e:
            logger.warning(f"BAML structured generation failed, falling back to template: {str(e)}")
            
            # Fallback to template-based generation
            return self._generate_template_based_schema(context)
    
    def _build_generation_prompt(
        self,
        context: Dict[str, Any],
        related_schemas: List[RelatedSchema],
        user_responses: List[UserResponse]
    ) -> str:
        """Build the LLM prompt for schema generation"""
        
        prompt = f"""You are an expert knowledge graph schema designer. Generate a YAML ontology schema based on the user's requirements.

**User Requirements:**
- Use Case: {context.get('use_case', 'Not specified')}
- Domain: {context.get('domain', 'Not specified')}
- Data Sources: {context.get('data_sources', 'Not specified')}
- Key Entities: {context.get('key_entities', 'Not specified')}
- Relationships: {context.get('relationships', 'Not specified')}
- Query Patterns: {context.get('query_patterns', 'Not specified')}
- Data Complexity: {context.get('data_complexity', 'Not specified')}
- Data Volume: {context.get('data_volume', 'Not specified')}
- Temporal Requirements: {context.get('temporal_requirements', 'Not specified')}

**Schema Format Requirements:**
1. Use YAML format with version 0.1.0
2. Structure: entities -> EntityName -> properties/relationships
3. Properties must have: type, description, and optionally: unique, required, index
4. Relationships must have: target, and optionally properties
5. Property types: str, int, float, bool, date
6. Include meaningful descriptions for all properties and relationships
8. Entity names in CamelCase; Relationship names in UPPER_CASE; Property names in camelCase;
9. No bi-directional synonymous relationships. Just use one. For e.g: Instead of both `INVOLVES` and `INVOLVED_BY`, just use one which makes more sense.

**Related Schemas for Reference:**
"""
        
        for schema in related_schemas[:3]:  # Include top 3 related schemas
            prompt += f"\n- {schema.title}: {schema.description} (similarity: {schema.similarity:.2f})"
        
        prompt += """

**Instructions:**
1. Create entities based on the key entities mentioned
2. Define properties for each entity based on the use case
3. Create relationships that support the query patterns
4. Add temporal fields if temporal requirements are specified
5. Include audit fields (created_at, updated_at) for entities with high data volume
6. Ensure the schema supports the specified use case effectively

**Output Format:**
Return ONLY the YAML schema content starting with 'version: 0.1.0' and 'entities:'.
DO NOT wrap the schema in any additional keys like 'ontology:' or any other wrapper.
The response should start directly with 'version: 0.1.0'.

Example of correct format:
version: 0.1.0
entities:
  EntityName:
    properties: ...

CONFIDENCE: [0.0-1.0]
SUGGESTIONS: [comma-separated list of suggestions]

Generate the schema now:"""
        
        return prompt
    
    def _build_refinement_prompt(
        self,
        current_schema: str,
        user_feedback: str,
        context: Dict[str, Any]
    ) -> str:
        """Build the LLM prompt for schema refinement"""
        
        return f"""You are refining a knowledge graph schema based on user feedback.

**Current Schema:**
```yaml
{current_schema}
```

**User Feedback:**
{user_feedback}

**Context:**
{context}

**Instructions:**
1. Analyze the user's feedback carefully
2. Make the requested changes while maintaining schema integrity
3. Preserve existing structure unless explicitly asked to change
4. Ensure all changes align with knowledge graph best practices
5. Maintain YAML format and property types

**Output Format:**
Return ONLY the YAML schema content starting with 'version: 0.1.0' and 'entities:'.
DO NOT wrap the schema in any additional keys like 'ontology:' or any other wrapper.
The response should start directly with 'version: 0.1.0'.

Example of correct format:
version: 0.1.0
entities:
  EntityName:
    properties: ...

CHANGES_MADE: [list of specific changes made]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [brief explanation of the changes]

Refined schema:"""
    
    def _parse_generation_response(
        self, 
        response_text: str, 
        context: Dict[str, Any]
    ) -> Tuple[str, float, List[str]]:
        """Parse LLM response for schema generation"""
        
        try:
            # Split response into schema and metadata
            parts = response_text.split('CONFIDENCE:')
            schema_content = parts[0].strip()
            
            # Remove any markdown formatting
            if schema_content.startswith('```yaml'):
                schema_content = schema_content[7:]
            if schema_content.endswith('```'):
                schema_content = schema_content[:-3]
            schema_content = schema_content.strip()
            
            # Validate YAML
            yaml.safe_load(schema_content)
            
            # Extract confidence
            confidence = 0.8  # Default
            if len(parts) > 1:
                conf_line = parts[1].split('\n')[0].strip()
                try:
                    confidence = float(conf_line)
                except:
                    pass
            
            # Extract suggestions
            suggestions = []
            if 'SUGGESTIONS:' in response_text:
                sugg_part = response_text.split('SUGGESTIONS:')[1].strip()
                suggestions = [s.strip() for s in sugg_part.split(',') if s.strip()]
            
            return schema_content, confidence, suggestions
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response, using fallback: {str(e)}")
            return self._generate_template_based_schema(context)
    
    def _parse_refinement_response(self, response_text: str) -> Tuple[str, List[str], float, str]:
        """Parse LLM response for schema refinement"""
        
        try:
            # Extract schema content
            schema_content = response_text.split('CHANGES_MADE:')[0].strip()
            if schema_content.startswith('```yaml'):
                schema_content = schema_content[7:]
            if schema_content.endswith('```'):
                schema_content = schema_content[:-3]
            schema_content = schema_content.strip()
            
            # Validate YAML
            yaml.safe_load(schema_content)
            
            # Extract changes made
            changes_made = []
            if 'CHANGES_MADE:' in response_text:
                changes_part = response_text.split('CHANGES_MADE:')[1]
                if 'CONFIDENCE:' in changes_part:
                    changes_part = changes_part.split('CONFIDENCE:')[0]
                changes_made = [c.strip() for c in changes_part.strip().split('\n') if c.strip()]
            
            # Extract confidence
            confidence = 0.8
            if 'CONFIDENCE:' in response_text:
                conf_part = response_text.split('CONFIDENCE:')[1]
                if 'EXPLANATION:' in conf_part:
                    conf_part = conf_part.split('EXPLANATION:')[0]
                try:
                    confidence = float(conf_part.strip())
                except:
                    pass
            
            # Extract explanation
            explanation = "Schema refined based on user feedback"
            if 'EXPLANATION:' in response_text:
                explanation = response_text.split('EXPLANATION:')[1].strip()
            
            return schema_content, changes_made, confidence, explanation
            
        except Exception as e:
            logger.error(f"Failed to parse refinement response: {str(e)}")
            raise
    
    def _generate_template_based_schema(self, context: Dict[str, Any]) -> Tuple[str, float, List[str]]:
        """Generate schema using templates as fallback"""
        
        domain = context.get('domain', 'Business/Enterprise')
        template = self.default_templates.get(domain, self.default_templates['Business/Enterprise'])
        
        # Basic customization based on context
        customized_schema = template.replace(
            '# Generated schema for: {{use_case}}',
            f'# Generated schema for: {context.get("use_case", "Not specified")}'
        )
        
        suggestions = [
            "Consider adding audit fields for better data tracking",
            "Review property types to ensure they match your data",
            "Add validation constraints for critical fields",
            "Consider temporal tracking if needed"
        ]
        
        return customized_schema, 0.7, suggestions
    
    def _load_default_templates(self) -> Dict[str, str]:
        """Load default schema templates for different domains"""
        
        return {
            'Business/Enterprise': """version: 0.1.0
# Generated schema for: {{use_case}}
entities:
  Organization:
    properties:
      name:
        type: str
        description: Organization name
        unique: true
        required: true
      type:
        type: str
        description: Organization type
        required: true
      industry:
        type: str
        description: Industry sector
        index: true
      created_at:
        type: date
        description: Record creation timestamp
        required: true
    relationships:
      HAS_EMPLOYEE:
        target: Person
        properties:
          role:
            type: str
            description: Employee role
          start_date:
            type: date
            description: Employment start date
      PARTNERED_WITH:
        target: Organization

  Person:
    properties:
      name:
        type: str
        description: Person's full name
        required: true
      email:
        type: str
        description: Email address
        unique: true
        index: true
      created_at:
        type: date
        description: Record creation timestamp
        required: true
    relationships:
      WORKS_FOR:
        target: Organization
        properties:
          position:
            type: str
            description: Job position""",

            'Healthcare/Medical': """version: 0.1.0
# Generated schema for: {{use_case}}
entities:
  Patient:
    properties:
      patient_id:
        type: str
        description: Unique patient identifier
        unique: true
        required: true
      name:
        type: str
        description: Patient full name
        required: true
      date_of_birth:
        type: date
        description: Patient date of birth
        required: true
      created_at:
        type: date
        description: Record creation timestamp
        required: true
    relationships:
      HAS_CONDITION:
        target: MedicalCondition
        properties:
          diagnosed_date:
            type: date
            description: Date of diagnosis
      TREATED_BY:
        target: HealthcareProvider

  HealthcareProvider:
    properties:
      provider_id:
        type: str
        description: Unique provider identifier
        unique: true
        required: true
      name:
        type: str
        description: Provider full name
        required: true
      specialty:
        type: str
        description: Medical specialty
        required: true
    relationships:
      TREATS:
        target: Patient

  MedicalCondition:
    properties:
      condition_code:
        type: str
        description: Standard condition code
        unique: true
        required: true
      name:
        type: str
        description: Condition name
        required: true""",

            'Financial/Banking': """version: 0.1.0
# Generated schema for: {{use_case}}
entities:
  Customer:
    properties:
      customer_id:
        type: str
        description: Unique customer identifier
        unique: true
        required: true
      name:
        type: str
        description: Customer full name
        required: true
      email:
        type: str
        description: Customer email
        unique: true
        index: true
      created_at:
        type: date
        description: Account creation date
        required: true
    relationships:
      OWNS:
        target: Account
        properties:
          ownership_type:
            type: str
            description: Type of ownership

  Account:
    properties:
      account_number:
        type: str
        description: Account number
        unique: true
        required: true
      account_type:
        type: str
        description: Type of account
        required: true
      balance:
        type: float
        description: Current account balance
        required: true
      created_at:
        type: date
        description: Account creation date
        required: true
    relationships:
      BELONGS_TO:
        target: Customer
      HAS_TRANSACTION:
        target: Transaction

  Transaction:
    properties:
      transaction_id:
        type: str
        description: Unique transaction identifier
        unique: true
        required: true
      amount:
        type: float
        description: Transaction amount
        required: true
      transaction_date:
        type: date
        description: Date of transaction
        required: true"""
        }
    
    def _convert_structured_schema_to_yaml(self, structured_schema) -> str:
        """Convert BAML structured schema to proper YAML format"""
        try:
            # Build YAML dictionary from structured schema
            yaml_dict = {
                "version": structured_schema.version,
                "entities": {}
            }
            
            # Convert entities
            for entity_name, entity_data in structured_schema.entities.items():
                yaml_dict["entities"][entity_name] = {
                    "properties": {}
                }
                
                # Convert properties
                if entity_data.properties:
                    for prop_name, prop_data in entity_data.properties.items():
                        prop_dict = {
                            "type": prop_data.type,
                            "description": prop_data.description
                        }
                        
                        # Add optional fields only if they exist
                        if hasattr(prop_data, 'required') and prop_data.required is not None:
                            prop_dict["required"] = prop_data.required
                        if hasattr(prop_data, 'unique') and prop_data.unique is not None:
                            prop_dict["unique"] = prop_data.unique
                        if hasattr(prop_data, 'index') and prop_data.index is not None:
                            prop_dict["index"] = prop_data.index
                            
                        yaml_dict["entities"][entity_name]["properties"][prop_name] = prop_dict
                
                # Convert relationships - CRITICAL: nested inside entities
                if entity_data.relationships:
                    yaml_dict["entities"][entity_name]["relationships"] = {}
                    
                    for rel_name, rel_data in entity_data.relationships.items():
                        rel_dict = {
                            "target": rel_data.target
                        }
                        
                        # Add relationship properties if they exist
                        if hasattr(rel_data, 'properties') and rel_data.properties:
                            rel_dict["properties"] = {}
                            for rel_prop_name, rel_prop_data in rel_data.properties.items():
                                rel_prop_dict = {
                                    "type": rel_prop_data.type,
                                    "description": rel_prop_data.description
                                }
                                if hasattr(rel_prop_data, 'required') and rel_prop_data.required is not None:
                                    rel_prop_dict["required"] = rel_prop_data.required
                                    
                                rel_dict["properties"][rel_prop_name] = rel_prop_dict
                        
                        yaml_dict["entities"][entity_name]["relationships"][rel_name] = rel_dict
            
            # Convert to YAML with proper formatting
            yaml_output = yaml.dump(
                yaml_dict, 
                default_flow_style=False, 
                sort_keys=False,
                indent=2,
                allow_unicode=True
            )
            
            return yaml_output
            
        except Exception as e:
            logger.error(f"Error converting structured schema to YAML: {str(e)}")
            # Return a basic template as fallback
            return self._generate_template_based_schema({})[0]


# Global service instance
schema_generation_service = SchemaGenerationService()