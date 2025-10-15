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
]


def get_full_text_index_name(entity_name: str) -> str:
    return f"_grf_fti_{entity_name.lower()}"
