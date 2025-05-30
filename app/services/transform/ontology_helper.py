from typing import Dict, List, Any, Type, Optional
import yaml
from datetime import datetime, timezone
from app.utils.constants import get_full_text_index_name
from pydantic import BaseModel, create_model, Field
from pathlib import Path
from typing import Union
from app.config import settings
from app.services.storage.neo4j import Neo4jStorage
from app.services.user_db_service import UserDatabaseService

class OntologyParser:
    """Parser for YAML ontology definitions"""

    def __init__(self, yaml_path: Union[str, Path]):
        """Initialize parser with YAML ontology"""
        # Load YAML content
        if isinstance(yaml_path, Path):
            with open(yaml_path) as f:
                yaml_content = f.read()
        else:
            yaml_content = yaml_path
        self.parsed_ontology = yaml.safe_load(yaml_content)
        self.ontology_yaml = yaml_content
        self.graph_model = self.build_graph_model()
        self.entities_only_model = self.build_entities_only_model()
        self.relationships_only_model = self.build_relationships_only_model()
        self.validate_ontology_structure()

    def validate_ontology_structure(self) -> None:
        """Validate ontology has required structure"""
        required_keys = ['version', 'entities']
        if not all(key in self.parsed_ontology for key in required_keys):
            raise ValueError(f"Ontology missing required keys: {required_keys}")

    def build_graph_model(self) -> Type[BaseModel]:
        """
        Build a complete Pydantic model structure from a YAML ontology definition.
        Returns a KnowledgeGraph class that can be used as a response_model.
        """
        ontology = self.parsed_ontology
        entities = ontology.get('entities', {})

        entity_models = {}
        relationship_models = {}

        # Create entity models with an id field
        for entity_name, entity_def in entities.items():
            props = entity_def.get('properties', {})
            field_definitions = {
                'id': (Optional[str], Field(default=None, description="Unique identifier for the entity"))
            }
            for prop_name, prop_def in props.items():
                field_type = self._get_field_type(prop_def.get('type', 'str'))
                is_required = prop_def.get('required', False)
                default_value = None if not is_required else ...
                field_definitions[prop_name] = (
                    Optional[field_type] if not is_required else field_type,
                    Field(
                        default=default_value,
                        description=prop_def.get('description', ''),
                        title=prop_name
                    )
                )
            entity_model = create_model(
                entity_name,
                __base__=BaseModel,
                __domain__="graphora",
                **field_definitions
            )
            entity_models[entity_name] = entity_model
            globals()[entity_name] = entity_model

        # Create relationship models with full SOURCE_TYPE_RELATION_TARGET_TYPE keys
        for entity_name, entity_def in entities.items():
            relationships = entity_def.get('relationships', {})
            entity_relationship_models = {}
            for rel_name, rel_def in relationships.items():
                target_name = rel_def.get('target')
                if target_name not in entity_models:
                    continue
                rel_props = rel_def.get('properties', {})
                rel_field_defs = {}
                for prop_name, prop_def in rel_props.items():
                    field_type = self._get_field_type(prop_def.get('type', 'str'))
                    is_required = prop_def.get('required', False)
                    default_value = None if not is_required else ...
                    rel_field_defs[prop_name] = (
                        Optional[field_type] if not is_required else field_type,
                        Field(
                            default_value,
                            description=prop_def.get('description', ''),
                            title=prop_name
                        )
                    )
                rel_property_model_name = f"{entity_name}_{rel_name}_Properties"
                rel_property_model = create_model(
                    rel_property_model_name,
                    __base__=BaseModel,
                    **rel_field_defs
                ) if rel_field_defs else None
                if rel_property_model:
                    globals()[rel_property_model_name] = rel_property_model

                # Use full SOURCE_TYPE_RELATION_TARGET_TYPE format
                rel_model_name = f"{entity_name}_{rel_name}_{target_name}_Relationship"
                if rel_property_model:
                    rel_model = create_model(
                        rel_model_name,
                        __base__=BaseModel,
                        __domain__="graphora",
                        source_id=(str, Field(..., description="ID of the source entity")),
                        target_id=(str, Field(..., description="ID of the target entity")),
                        properties=(Optional[rel_property_model], None)
                    )
                else:
                    rel_model = create_model(
                        rel_model_name,
                        __base__=BaseModel,
                        __domain__="graphora",
                        source_id=(str, Field(..., description="ID of the source entity")),
                        target_id=(str, Field(..., description="ID of the target entity"))
                    )
                globals()[rel_model_name] = rel_model
                entity_relationship_models[rel_name] = rel_model

            if entity_relationship_models:
                relationship_models[entity_name] = entity_relationship_models

        # Create the KnowledgeGraph model with full relationship keys
        kg_fields = {
            "extraction_timestamp": (Optional[str], Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())),
            "tokens_used": (Optional[int], None),
            "confidence_score": (Optional[float], None)
        }
        for entity_name, entity_model in entity_models.items():
            kg_fields[entity_name + "_list"] = (
                Optional[List[entity_model]],
                Field(default_factory=list, description=f"List of {entity_name} entities")
            )
        for source_name, rels in relationship_models.items():
            for rel_name, rel_model in rels.items():
                target_name = ontology['entities'][source_name]['relationships'][rel_name]['target']
                field_name = f"{source_name}_{rel_name}_{target_name}"
                kg_fields[field_name] = (
                    Optional[List[rel_model]],
                    Field(default_factory=list, description=f"Relationships of type {rel_name} from {source_name} to {target_name}")
                )

        KnowledgeGraph = create_model(
            "KnowledgeGraph",
            __base__=BaseModel,
            __domain__="graphora",
            **kg_fields
        )
        KnowledgeGraph.__entity_models__ = entity_models
        KnowledgeGraph.__relationship_models__ = relationship_models
        return KnowledgeGraph

    def _get_field_type(self, type_str: str) -> Type:
        """Convert string type to actual Python type."""
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': List[Any],
            'dict': Dict[str, Any],
        }
        return type_mapping.get(type_str, str)

    def build_entities_only_model(self) -> Type[BaseModel]:
        """Build a Pydantic model that only contains entity list fields with id."""
        full_model = self.graph_model
        entity_models = full_model.__entity_models__

        kg_fields = {
            "extraction_timestamp": (Optional[str], Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())),
            "tokens_used": (Optional[int], None),
            "confidence_score": (Optional[float], None)
        }
        for entity_name, entity_model in entity_models.items():
            kg_fields[entity_name + "_list"] = (
                Optional[List[entity_model]],
                Field(default_factory=list, description=f"List of {entity_name} entities")
            )

        EntitiesOnlyModel = create_model(
            "EntitiesOnlyModel",
            __base__=BaseModel,
            __domain__="graphora",
            **kg_fields
        )
        EntitiesOnlyModel.__entity_models__ = entity_models
        return EntitiesOnlyModel

    def build_relationships_only_model(self) -> Type[BaseModel]:
        """Build a Pydantic model that only contains relationship fields with source_id and target_id."""
        full_model = self.graph_model
        entity_models = full_model.__entity_models__
        relationship_models = full_model.__relationship_models__

        kg_fields = {
            "extraction_timestamp": (Optional[str], Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())),
            "tokens_used": (Optional[int], None),
            "confidence_score": (Optional[float], None)
        }
        for source_name, rels in relationship_models.items():
            for rel_name, rel_model in rels.items():
                target_name = self.parsed_ontology['entities'][source_name]['relationships'][rel_name]['target']
                field_name = f"{source_name}_{rel_name}_{target_name}"
                kg_fields[field_name] = (
                    Optional[List[rel_model]],
                    Field(default_factory=list, description=f"Relationships of type {rel_name} from {source_name} to {target_name}")
                )

        RelationshipsOnlyModel = create_model(
            "RelationshipsOnlyModel",
            __base__=BaseModel,
            __domain__="graphora",
            **kg_fields
        )
        # RelationshipsOnlyModel.__entity_models__ = entity_models
        RelationshipsOnlyModel.__relationship_models__ = relationship_models
        return RelationshipsOnlyModel
    
    
    async def build_full_text_indexes(self) -> None:
        """Build full text indexes for all entities and relationships defined in the ontology."""
        staging_storage = Neo4jStorage(
            uri=settings.STAGING_NEO4J_URI,
            username=settings.STAGING_NEO4J_USER,
            password=settings.STAGING_NEO4J_PASSWORD,
            database=settings.STAGING_NEO4J_DATABASE
        )
        prod_storage = Neo4jStorage(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DB
        )
        
        for entity_name, entity_def in self.parsed_ontology['entities'].items():
            # Create full text index for entity
            index_name = get_full_text_index_name(entity_name)
            
            # Get properties from ontology
            ontology_props = entity_def.get('properties', {})
            ontology_prop_names = [f'{prop}' for prop in ontology_props.keys()]
            
            # Try to get all properties from the database
            try:
                staging_props = await staging_storage.get_all_node_properties(entity_name)
                # Combine ontology properties with properties found in the database
                prop_names = list(set(ontology_prop_names + staging_props))
            except Exception as e:
                # If we can't get properties from the database, use ontology properties
                prop_names = ontology_prop_names
            
            # If we still don't have any properties, skip this entity
            if not prop_names:
                continue
                
            # Create indexes in both environments
            await staging_storage.create_or_replace_ft_index_for_node(index_name, entity_name, prop_names)
            await prod_storage.create_or_replace_ft_index_for_node(index_name, entity_name, prop_names)
            
        # Create full text index for relationships
        for source_name, rels in self.parsed_ontology['entities'].items():
            for rel_name, rel_def in rels.get('relationships', {}).items():
                target_name = rel_def.get('target', None)
                if not target_name:
                    continue
                    
                index_name = get_full_text_index_name(f"{source_name}_{rel_name}_{target_name}")
                
                # Get properties from ontology
                ontology_props = rel_def.get('properties', {})
                ontology_prop_names = [f'{prop}' for prop in ontology_props.keys()]
                
                # Try to get all properties from the database
                try:
                    staging_rel_props = await staging_storage.get_all_relationship_properties(rel_name)
                    # Combine ontology properties with properties found in the database
                    prop_names = list(set(ontology_prop_names + staging_rel_props))
                except Exception as e:
                    # If we can't get properties from the database, use ontology properties
                    prop_names = ontology_prop_names
                
                # If we still don't have any properties, skip this relationship
                if not prop_names:
                    continue
                    
                # Create indexes in both environments
                await staging_storage.create_or_replace_ft_index_for_relationship(
                    index_name, source_name, rel_name, target_name, prop_names
                )
                await prod_storage.create_or_replace_ft_index_for_relationship(
                    index_name, source_name, rel_name, target_name, prop_names
                )

    async def build_full_text_indexes_for_user(self, user_id: str) -> None:
        """Build full text indexes for all entities and relationships defined in the ontology for a specific user."""
        # Get user's database configurations
        user_config = await UserDatabaseService.get_user_config(user_id)
        
        # Create storage instances for user's databases
        staging_storage = Neo4jStorage(
            uri=user_config.stagingDb.uri,
            username=user_config.stagingDb.username,
            password=user_config.stagingDb.password,
            database="neo4j"  # Default database name
        )
        prod_storage = Neo4jStorage(
            uri=user_config.prodDb.uri,
            username=user_config.prodDb.username,
            password=user_config.prodDb.password,
            database="neo4j"  # Default database name
        )
        
        for entity_name, entity_def in self.parsed_ontology['entities'].items():
            # Create full text index for entity
            index_name = get_full_text_index_name(entity_name)
            
            # Get properties from ontology
            ontology_props = entity_def.get('properties', {})
            ontology_prop_names = [f'{prop}' for prop in ontology_props.keys()]
            
            # Try to get all properties from the database
            try:
                staging_props = await staging_storage.get_all_node_properties(entity_name)
                # Combine ontology properties with properties found in the database
                prop_names = list(set(ontology_prop_names + staging_props))
            except Exception as e:
                # If we can't get properties from the database, use ontology properties
                prop_names = ontology_prop_names
            
            # If we still don't have any properties, skip this entity
            if not prop_names:
                continue
                
            # Create indexes in both environments
            await staging_storage.create_or_replace_ft_index_for_node(index_name, entity_name, prop_names)
            await prod_storage.create_or_replace_ft_index_for_node(index_name, entity_name, prop_names)
            
        # Create full text index for relationships
        for source_name, rels in self.parsed_ontology['entities'].items():
            for rel_name, rel_def in rels.get('relationships', {}).items():
                target_name = rel_def.get('target', None)
                if not target_name:
                    continue
                    
                index_name = get_full_text_index_name(f"{source_name}_{rel_name}_{target_name}")
                
                # Get properties from ontology
                ontology_props = rel_def.get('properties', {})
                ontology_prop_names = [f'{prop}' for prop in ontology_props.keys()]
                
                # Try to get all properties from the database
                try:
                    staging_rel_props = await staging_storage.get_all_relationship_properties(rel_name)
                    # Combine ontology properties with properties found in the database
                    prop_names = list(set(ontology_prop_names + staging_rel_props))
                except Exception as e:
                    # If we can't get properties from the database, use ontology properties
                    prop_names = ontology_prop_names
                
                # If we still don't have any properties, skip this relationship
                if not prop_names:
                    continue
                    
                # Create indexes in both environments
                await staging_storage.create_or_replace_ft_index_for_relationship(
                    index_name, source_name, rel_name, target_name, prop_names
                )
                await prod_storage.create_or_replace_ft_index_for_relationship(
                    index_name, source_name, rel_name, target_name, prop_names
                )