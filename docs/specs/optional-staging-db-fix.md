# Specification: Optional Staging Database Support

## Overview

The staging database is now optional. When not configured, the system should fall back to in-memory storage for staging operations. This document specifies all the changes needed to support this across the codebase.

## Problem Statement

After making `stagingDb` optional in the config schema, multiple places in the codebase still assume it exists:

```
'NoneType' object has no attribute 'uri'
```

This occurs because code like `user_config.stagingDb.uri` fails when `stagingDb` is `None`.

## Design Principles

1. **Staging DB is optional** - Falls back to in-memory storage
2. **Production DB required for merge only** - Merge operations should fail early with clear error if no prod DB
3. **Graceful degradation** - Features should work with reduced functionality rather than crash
4. **Clear user feedback** - Users should know when they're using in-memory mode

---

## Critical Fixes Required

### 1. Storage Factory (`app/services/storage/factory.py`)

**Current (Line 110):**
```python
db_config = user_config.stagingDb if use_staging else user_config.productionDb
```

**Fix:**
```python
async def create_storage_for_user(
    user_id: str,
    use_staging: bool = True,
) -> GraphStorageInterface:
    """Create storage for a user based on their configuration.

    If use_staging=True and stagingDb is not configured, falls back to in-memory.
    If use_staging=False (production) and prodDb is not configured, raises ValueError.
    """
    storage_type = settings.STORAGE_TYPE.lower()

    if storage_type == "memory":
        from app.services.storage.memory import InMemoryStorage
        logger.info(f"Using in-memory storage for user {user_id}")
        return InMemoryStorage(user_id=user_id)

    elif storage_type == "neo4j":
        from app.services.user_db_service import UserDatabaseService
        user_config = await UserDatabaseService.get_user_config(user_id)

        if use_staging:
            # Staging is optional - fall back to memory if not configured
            if user_config.stagingDb is None:
                from app.services.storage.memory import InMemoryStorage
                logger.info(f"No staging DB configured for {user_id}, using in-memory storage")
                return InMemoryStorage(user_id=user_id)
            db_config = user_config.stagingDb
        else:
            # Production is required for merge operations
            if user_config.prodDb is None:
                raise ValueError(
                    "Production database is required for merge operations. "
                    "Please configure a production database in settings."
                )
            db_config = user_config.prodDb

        from app.services.storage.neo4j import Neo4jStorage
        logger.info(f"Using Neo4j storage at {db_config.uri} for user {user_id}")
        return Neo4jStorage(
            uri=db_config.uri,
            username=db_config.username,
            password=db_config.password,
            database="neo4j",
        )
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")
```

**Add helper function:**
```python
async def get_storage_type_for_user(user_id: str, use_staging: bool = True) -> str:
    """Determine what storage type will be used for a user.

    Returns: 'neo4j' or 'memory'
    """
    if settings.STORAGE_TYPE.lower() == "memory":
        return "memory"

    from app.services.user_db_service import UserDatabaseService
    user_config = await UserDatabaseService.get_user_config(user_id)

    if use_staging and user_config.stagingDb is None:
        return "memory"
    return "neo4j"
```

---

### 2. Quality API (`app/api/quality.py` and `app/api/dashboard.py`)

**Current pattern:**
```python
neo4j_storage = Neo4jStorage(
    uri=user_config.stagingDb.uri,
    ...
)
```

**Fix - Create helper function:**
```python
async def _get_staging_storage(user_id: str) -> GraphStorageInterface:
    """Get staging storage for quality operations.

    Falls back to in-memory if staging DB is not configured.
    """
    from app.services.storage.factory import create_storage_for_user
    return await create_storage_for_user(user_id, use_staging=True)
```

**Update all quality endpoints to use this helper instead of direct Neo4j instantiation.**

---

### 3. Quality Tasks (`app/services/quality/tasks.py`)

**Lines 57-60 and 117-121 - Replace direct Neo4j usage:**

```python
# Before
storage = Neo4jStorage(
    uri=user_config.stagingDb.uri,
    ...
)

# After
from app.services.storage.factory import create_storage_for_user
storage = await create_storage_for_user(user_id, use_staging=True)
```

---

### 4. Ontology Helper (`app/services/transform/ontology_helper.py`)

**Lines 446-534 - `build_full_text_indexes_for_user()`:**

Full-text indexes only make sense for Neo4j. Skip for in-memory storage.

```python
async def build_full_text_indexes_for_user(self, user_id: str) -> None:
    """Build full text indexes for entities/relationships in the ontology.

    Indexes are only created for configured Neo4j databases.
    Skips staging if not configured, skips production if not configured.
    """
    user_config = await UserDatabaseService.get_user_config(user_id)

    staging_storage = None
    prod_storage = None

    # Create storage for staging if configured
    if user_config.stagingDb is not None:
        from app.services.storage.neo4j import Neo4jStorage
        staging_storage = Neo4jStorage(
            uri=user_config.stagingDb.uri,
            username=user_config.stagingDb.username,
            password=user_config.stagingDb.password,
            database="neo4j",
        )
        logger.info(f"Will create indexes on staging DB for user {user_id}")
    else:
        logger.info(f"No staging DB configured for {user_id}, skipping staging indexes")

    # Create storage for production if configured
    if user_config.prodDb is not None:
        from app.services.storage.neo4j import Neo4jStorage
        prod_storage = Neo4jStorage(
            uri=user_config.prodDb.uri,
            username=user_config.prodDb.username,
            password=user_config.prodDb.password,
            database="neo4j",
        )
        logger.info(f"Will create indexes on production DB for user {user_id}")
    else:
        logger.info(f"No production DB configured for {user_id}, skipping production indexes")

    # If neither DB is configured, nothing to do
    if staging_storage is None and prod_storage is None:
        logger.info(f"No databases configured for {user_id}, skipping all index creation")
        return

    # Create indexes for each entity...
    for entity_name, entity_def in self.parsed_ontology["entities"].items():
        # ... existing logic but check storage before using:
        if staging_storage:
            await staging_storage.create_or_replace_ft_index_for_node(...)
        if prod_storage:
            await prod_storage.create_or_replace_ft_index_for_node(...)
```

---

### 5. Merge Flow (`app/services/merge/new_merger.py`)

**Line 576-578 - `_extract_staging_graph()`:**

The merge flow is special - it needs to read from staging to merge into production.

**Options:**
1. If staging DB is configured → Read from Neo4j
2. If staging DB is NOT configured (in-memory mode) → Read from in-memory storage

```python
async def _extract_staging_graph(self, transform_id: str) -> Tuple[List[Node], List[Edge]]:
    """Extract the staging graph for a transform.

    Reads from staging Neo4j if configured, otherwise from in-memory storage.
    """
    from app.services.storage.factory import create_storage_for_user

    storage = await create_storage_for_user(self.user_id, use_staging=True)

    # The rest of the extraction logic remains the same
    # storage interface is the same for both Neo4j and InMemory
    ...
```

**However, for merge to production to work, production DB MUST be configured.**
Add validation at merge start:

```python
async def start_merge(self, transform_id: str) -> MergeResult:
    """Start a merge operation."""
    # Validate production DB is configured
    user_config = await UserDatabaseService.get_user_config(self.user_id)
    if user_config.prodDb is None:
        raise ValueError(
            "Production database is required for merge operations. "
            "Please configure a production database before merging."
        )

    # Continue with merge...
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure (Do First)

- [ ] Update `app/services/storage/factory.py`
  - [ ] Add null check for stagingDb, fall back to InMemoryStorage
  - [ ] Add explicit error for missing prodDb when use_staging=False
  - [ ] Add `get_storage_type_for_user()` helper

### Phase 2: Quality Operations

- [ ] Update `app/api/quality.py`
  - [ ] Replace direct Neo4jStorage with factory function
  - [ ] Handle in-memory storage case

- [ ] Update `app/api/dashboard.py`
  - [ ] Same changes as quality.py

- [ ] Update `app/services/quality/tasks.py`
  - [ ] Replace direct Neo4jStorage with factory function
  - [ ] Both `quality_validation_task()` and `auto_approval_check_task()`

### Phase 3: Ontology & Indexing

- [ ] Update `app/services/transform/ontology_helper.py`
  - [ ] `build_full_text_indexes_for_user()` - skip if DB not configured
  - [ ] Log clearly when skipping index creation

### Phase 4: Merge Operations

- [ ] Update `app/services/merge/new_merger.py`
  - [ ] `_extract_staging_graph()` - use factory function
  - [ ] Add validation at merge start for production DB requirement

### Phase 5: Testing

- [ ] Add tests for in-memory fallback scenarios
- [ ] Add tests for merge validation
- [ ] Test full workflow: upload → transform → (no merge without prod DB)

---

## Error Messages

Standardize error messages for clarity:

| Scenario | Error Message |
|----------|---------------|
| Merge without prod DB | "Production database is required for merge operations. Please configure a production database in Settings → Databases." |
| Index creation skipped | (Log only) "No staging/production database configured, skipping full-text index creation" |

---

## API Response Considerations

When returning configuration status, include storage mode:

```json
{
  "staging_storage_type": "memory",  // or "neo4j"
  "production_storage_type": null,   // null means not configured
  "can_merge": false,
  "merge_blocked_reason": "Production database not configured"
}
```

---

## Testing Scenarios

1. **User with no databases configured**
   - Transform: Should work (in-memory staging)
   - Quality: Should work (in-memory)
   - Merge: Should fail with clear error

2. **User with only staging DB**
   - Transform: Should work (Neo4j staging)
   - Quality: Should work
   - Merge: Should fail with clear error

3. **User with only production DB**
   - Transform: Should work (in-memory staging)
   - Quality: Should work (in-memory)
   - Merge: Should work (read from memory, write to Neo4j prod)

4. **User with both databases**
   - Everything should work as before
