from typing import Dict, List, Any, Type, Optional
import yaml
from datetime import datetime, timezone
from pydantic import BaseModel, create_model, Field
from pathlib import Path
from typing import Union

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