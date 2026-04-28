from typing import Dict, List, Any, Type, Optional
import yaml
import logging
from datetime import datetime, timezone
from graphora_server.utils.constants import get_full_text_index_name
from pydantic import BaseModel, create_model, Field
from pathlib import Path
from typing import Union
from graphora_server.services.user_db_service import UserDatabaseService
from graphora_server.db import postgres as db

logger = logging.getLogger(__name__)


class OntologyParser:
    """Parser for YAML ontology definitions"""

    def __init__(self, yaml_path: Union[str, Path], user_id: Optional[str] = None):
        """Initialize parser with YAML ontology

        Args:
            yaml_path: Path to YAML file or ontology ID or YAML content string
            user_id: User ID for database fallback (optional)
        """
        # Load YAML content with database fallback
        yaml_content = self._load_yaml_content(yaml_path, user_id)
        self.parsed_ontology = yaml.safe_load(yaml_content)
        self.ontology_yaml = yaml_content
        self.graph_model = self.build_graph_model()
        self.entities_only_model = self.build_entities_only_model()
        self.relationships_only_model = self.build_relationships_only_model()
        self.validate_ontology_structure()

    def _load_yaml_content(
        self, yaml_path: Union[str, Path], user_id: Optional[str] = None
    ) -> str:
        """Load YAML content from file, database, or string with enhanced fallback logic"""

        # If it's already YAML content (string), return it
        if isinstance(yaml_path, str) and not Path(yaml_path).exists():
            # Check if it looks like YAML content
            if "version:" in yaml_path and "entities:" in yaml_path:
                return yaml_path

        # Try to load from file path first
        file_path = Path(yaml_path) if isinstance(yaml_path, str) else yaml_path

        # Check if file exists locally
        if file_path.exists():
            try:
                with open(file_path) as f:
                    content = f.read()
                    # Validate that it's valid YAML content
                    yaml.safe_load(content)
                    return content
            except (IOError, yaml.YAMLError) as e:
                logger.warning(
                    f"Warning: Failed to read local ontology file {file_path}: {e}"
                )
                # Continue to database fallback if local file is corrupted

        # If file doesn't exist or is corrupted and we have user_id, try database
        if user_id:
            # Extract ontology ID from filename if it's a path, otherwise use as-is
            if isinstance(yaml_path, (str, Path)):
                file_path = Path(yaml_path)
                if file_path.suffix == ".yaml":
                    ontology_id = file_path.stem
                else:
                    ontology_id = str(yaml_path)
            else:
                ontology_id = str(yaml_path)

            logger.info(
                f"Attempting to load ontology '{ontology_id}' from database for user {user_id}"
            )
            db_content = self._load_from_database(ontology_id, user_id)
            if db_content:
                logger.info(
                    f"Successfully loaded ontology '{ontology_id}' from database"
                )
                return db_content
            else:
                logger.info(
                    f"Ontology '{ontology_id}' not found in database for user {user_id}"
                )

        # If we still couldn't load the ontology, provide a helpful error message
        error_msg = f"Ontology not found: {yaml_path}"
        if file_path.exists():
            error_msg += " (local file exists but is corrupted)"
        else:
            error_msg += " (local file not found)"

        if user_id:
            error_msg += f" and not found in database for user {user_id}"
        else:
            error_msg += " and no user_id provided for database fallback"

        raise FileNotFoundError(error_msg)

    def _load_from_database(self, ontology_id: str, user_id: str) -> Optional[str]:
        """Load ontology content from Postgres."""
        try:
            record = db.sync_fetchrow(
                """
                SELECT yaml_content
                FROM ontologies
                WHERE id = %s AND user_id = %s AND is_active = TRUE
                """,
                ontology_id,
                user_id,
            )

            if record and record.get("yaml_content"):
                yaml_content = record["yaml_content"]
                try:
                    yaml.safe_load(yaml_content)
                    return yaml_content
                except yaml.YAMLError as e:
                    logger.warning(
                        "Warning: Ontology '%s' from database contains invalid YAML: %s",
                        ontology_id,
                        e,
                    )
                    return None

            logger.info(
                "No active ontology found with id '%s' for user '%s'",
                ontology_id,
                user_id,
            )
            return None

        except Exception as e:
            logger.error(
                "Error loading ontology '%s' from database: %s", ontology_id, e
            )
            return None

    def validate_ontology_structure(self) -> None:
        """Validate ontology has required structure"""
        required_keys = ["version", "entities"]
        if not all(key in self.parsed_ontology for key in required_keys):
            raise ValueError(f"Ontology missing required keys: {required_keys}")

    def build_graph_model(self) -> Type[BaseModel]:
        """
        Build a complete Pydantic model structure from a YAML ontology definition.
        Returns a KnowledgeGraph class that can be used as a response_model.
        """
        ontology = self.parsed_ontology
        entities = ontology.get("entities", {})

        entity_models = {}
        relationship_models = {}

        # Create entity models with an id field
        for entity_name, entity_def in entities.items():
            props = entity_def.get("properties", {})
            field_definitions = {
                "id": (
                    Optional[str],
                    Field(default=None, description="Unique identifier for the entity"),
                )
            }
            for prop_name, prop_def in props.items():
                field_type = self._get_field_type(prop_def.get("type", "str"))
                is_required = prop_def.get("required", False)
                default_value = None if not is_required else ...
                field_definitions[prop_name] = (
                    Optional[field_type] if not is_required else field_type,
                    Field(
                        default=default_value,
                        description=prop_def.get("description", ""),
                        title=prop_name,
                    ),
                )
            entity_model = create_model(
                entity_name,
                __base__=BaseModel,
                __domain__="graphora",
                **field_definitions,
            )
            entity_models[entity_name] = entity_model
            globals()[entity_name] = entity_model

        # Create relationship models with full SOURCE_TYPE_RELATION_TARGET_TYPE keys
        for entity_name, entity_def in entities.items():
            relationships = entity_def.get("relationships", {})
            entity_relationship_models = {}
            for rel_name, rel_def in relationships.items():
                target_name = rel_def.get("target")
                if target_name not in entity_models:
                    continue
                rel_props = rel_def.get("properties", {})
                rel_field_defs = {}
                for prop_name, prop_def in rel_props.items():
                    field_type = self._get_field_type(prop_def.get("type", "str"))
                    is_required = prop_def.get("required", False)
                    default_value = None if not is_required else ...
                    rel_field_defs[prop_name] = (
                        Optional[field_type] if not is_required else field_type,
                        Field(
                            default_value,
                            description=prop_def.get("description", ""),
                            title=prop_name,
                        ),
                    )
                rel_property_model_name = f"{entity_name}_{rel_name}_Properties"
                rel_property_model = (
                    create_model(
                        rel_property_model_name, __base__=BaseModel, **rel_field_defs
                    )
                    if rel_field_defs
                    else None
                )
                if rel_property_model:
                    globals()[rel_property_model_name] = rel_property_model

                # Use full SOURCE_TYPE_RELATION_TARGET_TYPE format
                rel_model_name = f"{entity_name}_{rel_name}_{target_name}_Relationship"
                if rel_property_model:
                    rel_model = create_model(
                        rel_model_name,
                        __base__=BaseModel,
                        __domain__="graphora",
                        source_id=(
                            str,
                            Field(..., description="ID of the source entity"),
                        ),
                        target_id=(
                            str,
                            Field(..., description="ID of the target entity"),
                        ),
                        properties=(Optional[rel_property_model], None),
                    )
                else:
                    rel_model = create_model(
                        rel_model_name,
                        __base__=BaseModel,
                        __domain__="graphora",
                        source_id=(
                            str,
                            Field(..., description="ID of the source entity"),
                        ),
                        target_id=(
                            str,
                            Field(..., description="ID of the target entity"),
                        ),
                    )
                globals()[rel_model_name] = rel_model
                entity_relationship_models[rel_name] = rel_model

            if entity_relationship_models:
                relationship_models[entity_name] = entity_relationship_models

        # Create the KnowledgeGraph model with full relationship keys
        kg_fields = {
            "extraction_timestamp": (
                Optional[str],
                Field(default_factory=lambda: datetime.now(timezone.utc).isoformat()),
            ),
            "tokens_used": (Optional[int], None),
            "confidence_score": (Optional[float], None),
        }
        for entity_name, entity_model in entity_models.items():
            kg_fields[entity_name + "_list"] = (
                Optional[List[entity_model]],
                Field(
                    default_factory=list, description=f"List of {entity_name} entities"
                ),
            )
        for source_name, rels in relationship_models.items():
            for rel_name, rel_model in rels.items():
                target_name = ontology["entities"][source_name]["relationships"][
                    rel_name
                ]["target"]
                field_name = f"{source_name}_{rel_name}_{target_name}"
                kg_fields[field_name] = (
                    Optional[List[rel_model]],
                    Field(
                        default_factory=list,
                        description=f"Relationships of type {rel_name} from {source_name} to {target_name}",
                    ),
                )

        KnowledgeGraph = create_model(
            "KnowledgeGraph", __base__=BaseModel, __domain__="graphora", **kg_fields
        )
        KnowledgeGraph.__entity_models__ = entity_models
        KnowledgeGraph.__relationship_models__ = relationship_models
        return KnowledgeGraph

    def _get_field_type(self, type_str: str) -> Type:
        """Convert string type to actual Python type."""
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": List[Any],
            "dict": Dict[str, Any],
        }
        return type_mapping.get(type_str, str)

    def build_entities_only_model(self) -> Type[BaseModel]:
        """Build a Pydantic model that only contains entity list fields with id."""
        full_model = self.graph_model
        entity_models = full_model.__entity_models__

        kg_fields = {
            "extraction_timestamp": (
                Optional[str],
                Field(default_factory=lambda: datetime.now(timezone.utc).isoformat()),
            ),
            "tokens_used": (Optional[int], None),
            "confidence_score": (Optional[float], None),
        }
        for entity_name, entity_model in entity_models.items():
            kg_fields[entity_name + "_list"] = (
                Optional[List[entity_model]],
                Field(
                    default_factory=list, description=f"List of {entity_name} entities"
                ),
            )

        EntitiesOnlyModel = create_model(
            "EntitiesOnlyModel", __base__=BaseModel, __domain__="graphora", **kg_fields
        )
        EntitiesOnlyModel.__entity_models__ = entity_models
        return EntitiesOnlyModel

    def build_relationships_only_model(self) -> Type[BaseModel]:
        """Build a Pydantic model that only contains relationship fields with source_id and target_id."""
        full_model = self.graph_model
        relationship_models = full_model.__relationship_models__

        kg_fields = {
            "extraction_timestamp": (
                Optional[str],
                Field(default_factory=lambda: datetime.now(timezone.utc).isoformat()),
            ),
            "tokens_used": (Optional[int], None),
            "confidence_score": (Optional[float], None),
        }
        for source_name, rels in relationship_models.items():
            for rel_name, rel_model in rels.items():
                target_name = self.parsed_ontology["entities"][source_name][
                    "relationships"
                ][rel_name]["target"]
                field_name = f"{source_name}_{rel_name}_{target_name}"
                kg_fields[field_name] = (
                    Optional[List[rel_model]],
                    Field(
                        default_factory=list,
                        description=f"Relationships of type {rel_name} from {source_name} to {target_name}",
                    ),
                )

        RelationshipsOnlyModel = create_model(
            "RelationshipsOnlyModel",
            __base__=BaseModel,
            __domain__="graphora",
            **kg_fields,
        )
        # RelationshipsOnlyModel.__entity_models__ = entity_models
        RelationshipsOnlyModel.__relationship_models__ = relationship_models
        return RelationshipsOnlyModel

    # DEPRECATED: Use build_full_text_indexes_for_user(user_id) instead
    # async def build_full_text_indexes(self) -> None:
    #     """Build full text indexes for all entities and relationships defined in the ontology."""
    #     staging_storage = Neo4jStorage(
    #         uri=settings.STAGING_NEO4J_URI,
    #         username=settings.STAGING_NEO4J_USER,
    #         password=settings.STAGING_NEO4J_PASSWORD,
    #         database=settings.STAGING_NEO4J_DATABASE
    #     )
    #     prod_storage = Neo4jStorage(
    #         uri=settings.NEO4J_URI,
    #         username=settings.NEO4J_USER,
    #         password=settings.NEO4J_PASSWORD,
    #         database=settings.NEO4J_DB
    #     )
    #
    #     for entity_name, entity_def in self.parsed_ontology['entities'].items():
    #         # Create full text index for entity
    #         index_name = get_full_text_index_name(entity_name)
    #
    #         # Get properties from ontology
    #         ontology_props = entity_def.get('properties', {})
    #         ontology_prop_names = [f'{prop}' for prop in ontology_props.keys()]
    #
    #         # Try to get all properties from the database
    #         try:
    #             staging_props = await staging_storage.get_all_node_properties(entity_name)
    #             # Combine ontology properties with properties found in the database
    #             prop_names = list(set(ontology_prop_names + staging_props))
    #         except Exception as e:
    #             # If we can't get properties from the database, use ontology properties
    #             prop_names = ontology_prop_names
    #
    #         # If we still don't have any properties, skip this entity
    #         if not prop_names:
    #             continue
    #
    #         # Create indexes in both environments
    #         await staging_storage.create_or_replace_ft_index_for_node(index_name, entity_name, prop_names)
    #         await prod_storage.create_or_replace_ft_index_for_node(index_name, entity_name, prop_names)
    #
    #     # Create full text index for relationships
    #     for source_name, rels in self.parsed_ontology['entities'].items():
    #         for rel_name, rel_def in rels.get('relationships', {}).items():
    #             target_name = rel_def.get('target', None)
    #             if not target_name:
    #                 continue
    #
    #             index_name = get_full_text_index_name(f"{source_name}_{rel_name}_{target_name}")
    #
    #             # Get properties from ontology
    #             ontology_props = rel_def.get('properties', {})
    #             ontology_prop_names = [f'{prop}' for prop in ontology_props.keys()]
    #
    #             # Try to get all properties from the database
    #             try:
    #                 staging_rel_props = await staging_storage.get_all_relationship_properties(rel_name)
    #                 # Combine ontology properties with properties found in the database
    #                 prop_names = list(set(ontology_prop_names + staging_rel_props))
    #             except Exception as e:
    #                 # If we can't get properties from the database, use ontology properties
    #                 prop_names = ontology_prop_names
    #
    #             # If we still don't have any properties, skip this relationship
    #             if not prop_names:
    #                 continue
    #
    #             # Create indexes in both environments
    #             await staging_storage.create_or_replace_ft_index_for_relationship(
    #                 index_name, source_name, rel_name, target_name, prop_names
    #             )
    #             await prod_storage.create_or_replace_ft_index_for_relationship(
    #                 index_name, source_name, rel_name, target_name, prop_names
    #             )

    async def build_full_text_indexes_for_user(self, user_id: str) -> None:
        """Build full text indexes for all entities and relationships defined in the ontology for a specific user.

        Indexes are only created for configured Neo4j databases.
        - Skips staging indexes if staging DB is not configured
        - Skips production indexes if production DB is not configured
        """
        import logging

        logger = logging.getLogger(__name__)

        # Route to whichever backend STORAGE_TYPE selects. Pre-slice-8
        # this method hard-wired Neo4jStorage and ignored
        # STORAGE_TYPE=postgres entirely, so the AGE adapter's GIN
        # polyfill (slice 8) wasn't on a live codepath. The slice-8
        # review caught that — this dispatch puts both staging and
        # prod through the configured backend.
        from graphora_server.config import settings

        storage_type = (settings.STORAGE_TYPE or "neo4j").lower()
        staging_storage = None
        prod_storage = None

        if storage_type == "memory":
            # In-memory storage has no full-text index concept;
            # skip entirely. Returns silently rather than calling
            # InMemoryStorage's no-op so we don't construct a
            # transient instance per ontology change.
            logger.info(
                "In-memory storage mode: skipping full-text index "
                "creation for user %s",
                user_id,
            )
            return

        if storage_type == "postgres":
            # Per-user Postgres routing (parallel to Neo4j's
            # stagingDb / prodDb shape) is slice 9+ work — for now
            # both staging and prod share the same global
            # POSTGRES_AGE_DSN, mirroring the factory's
            # create_storage_for_user behaviour. The merge contract
            # ("indexes exist on both staging and prod") is
            # preserved because the same adapter satisfies both
            # roles; idempotent DROP IF EXISTS + CREATE INDEX in
            # the AGE polyfill makes the duplicate calls safe.
            from graphora_server.services.storage.factory import _build_age_storage

            age_storage = _build_age_storage()
            staging_storage = age_storage
            prod_storage = age_storage
            logger.info(
                "Will create full-text indexes on Apache AGE backend "
                "for user %s (shared staging+prod until slice 9 wires "
                "per-user Postgres routing)",
                user_id,
            )
        else:
            # Neo4j path — preserve the existing dual-DB shape so
            # operators with separate staging/prod Neo4j clusters
            # keep working unchanged. Get user's database
            # configurations.
            user_config = await UserDatabaseService.get_user_config(user_id)

            # Create storage for staging if configured
            if user_config.stagingDb is not None:
                from graphora_server.services.storage.neo4j import Neo4jStorage

                staging_storage = Neo4jStorage(
                    uri=user_config.stagingDb.uri,
                    username=user_config.stagingDb.username,
                    password=user_config.stagingDb.password,
                    database="neo4j",
                )
                logger.info(
                    f"Will create full-text indexes on staging DB for user {user_id}"
                )
            else:
                logger.info(
                    f"No staging DB configured for user {user_id}, skipping staging indexes"
                )

            # Create storage for production if configured
            if user_config.prodDb is not None:
                from graphora_server.services.storage.neo4j import Neo4jStorage

                prod_storage = Neo4jStorage(
                    uri=user_config.prodDb.uri,
                    username=user_config.prodDb.username,
                    password=user_config.prodDb.password,
                    database="neo4j",
                )
                logger.info(
                    f"Will create full-text indexes on production DB for user {user_id}"
                )
            else:
                logger.info(
                    f"No production DB configured for user {user_id}, skipping production indexes"
                )

        # If neither DB is configured, nothing to do
        if staging_storage is None and prod_storage is None:
            logger.info(
                f"No databases configured for user {user_id}, skipping all index creation"
            )
            return

        for entity_name, entity_def in self.parsed_ontology["entities"].items():
            # Create full text index for entity
            index_name = get_full_text_index_name(entity_name)

            # Get properties from ontology
            ontology_props = entity_def.get("properties", {})
            ontology_prop_names = [f"{prop}" for prop in ontology_props.keys()]

            # Try to get all properties from the database (prefer staging if available)
            prop_names = ontology_prop_names
            if staging_storage is not None:
                try:
                    staging_props = await staging_storage.get_all_node_properties(
                        entity_name
                    )
                    # Combine ontology properties with properties found in the database
                    prop_names = list(set(ontology_prop_names + staging_props))
                except Exception:
                    # If we can't get properties from the database, use ontology properties
                    pass
            elif prod_storage is not None:
                try:
                    prod_props = await prod_storage.get_all_node_properties(entity_name)
                    prop_names = list(set(ontology_prop_names + prod_props))
                except Exception:
                    pass

            # If we still don't have any properties, skip this entity
            if not prop_names:
                continue

            # Create indexes only for configured databases
            if staging_storage is not None:
                await staging_storage.create_or_replace_ft_index_for_node(
                    index_name, entity_name, prop_names
                )
            if prod_storage is not None:
                await prod_storage.create_or_replace_ft_index_for_node(
                    index_name, entity_name, prop_names
                )

        # Create full text index for relationships
        for source_name, rels in self.parsed_ontology["entities"].items():
            for rel_name, rel_def in rels.get("relationships", {}).items():
                target_name = rel_def.get("target", None)
                if not target_name:
                    continue

                index_name = get_full_text_index_name(
                    f"{source_name}_{rel_name}_{target_name}"
                )

                # Get properties from ontology
                ontology_props = rel_def.get("properties", {})
                ontology_prop_names = [f"{prop}" for prop in ontology_props.keys()]

                # Try to get all properties from the database (prefer staging if available)
                prop_names = ontology_prop_names
                if staging_storage is not None:
                    try:
                        staging_rel_props = (
                            await staging_storage.get_all_relationship_properties(
                                rel_name
                            )
                        )
                        prop_names = list(set(ontology_prop_names + staging_rel_props))
                    except Exception:
                        pass
                elif prod_storage is not None:
                    try:
                        prod_rel_props = (
                            await prod_storage.get_all_relationship_properties(rel_name)
                        )
                        prop_names = list(set(ontology_prop_names + prod_rel_props))
                    except Exception:
                        pass

                # If we still don't have any properties, skip this relationship
                if not prop_names:
                    continue

                # Create indexes only for configured databases
                if staging_storage is not None:
                    await staging_storage.create_or_replace_ft_index_for_relationship(
                        index_name, source_name, rel_name, target_name, prop_names
                    )
                if prod_storage is not None:
                    await prod_storage.create_or_replace_ft_index_for_relationship(
                        index_name, source_name, rel_name, target_name, prop_names
                    )
