INTERNAL_NODE_TYPES = {"__Checkpoint__"}

VALID_FROM = "__valid_from"
VALID_TO = "__valid_to"
UPDATED = "__updated"
TRANSFORM_ID = "__tid"
MERGE_ID = "__mid"

PREVIOUS_VERSION_RELATIONSHIP_TYPE = "__PREV_VER"

SYSTEM_PROPERTIES = [
    "id",
    "confidence_score",
    "created_at",
    "updated_at",
    "extraction_timestamp",
    "provenance",
    "chunk_ids",
    TRANSFORM_ID,
    MERGE_ID,
    VALID_FROM,
    VALID_TO,
    UPDATED,
    # A1-prov source-span fields. Stamped onto every node/edge by
    # services/transform/helpers.py::_attach_provenance_properties
    # so the Explorer Evidence tab + MCP get_evidence can surface
    # them. Listing here keeps the rest of the system (similarity
    # scoring in find_similar_nodes, ontology-validation skip-lists
    # in services/quality/validator.py) from treating them as
    # user-meaningful signal — without this, two nodes from the
    # same document score artificially similar on source_text /
    # document_name overlap.
    "source_chunk",
    "source_chunk_id",
    "source_text",
    "source",
    "source_file",
    "document_id",
    "document_name",
    "chunk_offset",
    "page_number",
    "extraction_confidence",
    # B0-prov-extend decision-trail fields. Same reasoning — these
    # are LLM/extraction telemetry, not entity signal. Mirrors the
    # _EVIDENCE_KEYS contract in graphora_server/mcp/server.py.
    "extractor_model",
    "prompt_version",
    "validator_score",
]


def get_full_text_index_name(entity_name: str) -> str:
    return f"_grf_fti_{entity_name.lower()}"
