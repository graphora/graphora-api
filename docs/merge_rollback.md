# Graph Merge Rollback Functionality

This document describes the rollback functionality for the Graph Merge service, which allows reverting changes made during a merge operation.

## Overview

The rollback functionality provides a safety mechanism to revert changes made to the production graph during a merge operation. This is particularly useful in scenarios where:

1. A merge operation introduces unexpected data issues
2. Validation failures occur after changes have been applied
3. A merge needs to be cancelled after it has been partially or fully executed

## Features

### 1. Complete Rollback

A complete rollback reverts all changes made during a specific merge operation. This includes:
- Restoring all affected nodes to their pre-merge state
- Restoring all affected relationships to their pre-merge state
- Removing any newly created nodes or relationships
- Updating the merge status to CANCELLED

### 2. Partial Rollback

A partial rollback allows for selective reverting of changes, targeting specific entities:
- Only specified nodes and their relationships are restored
- Other changes from the merge remain intact
- Useful for targeted fixes without undoing all work

### 3. Automatic Rollback on Validation Failure

The system can automatically trigger a rollback when validation fails:
- Configurable via the `auto_rollback` parameter in validation
- Prevents invalid data from persisting in the production graph
- Provides detailed validation results including rollback information

## Implementation Details

### Snapshot Creation

Before applying changes during a merge, the system creates a snapshot of the affected production nodes and relationships. This snapshot includes:
- Node properties and labels
- Relationship properties and types
- The complete subgraph of affected entities

Snapshots are stored in Redis with a configurable TTL (Time To Live).

### Rollback Process

The rollback process follows these steps:
1. Load the snapshot for the specified merge
2. Begin a transaction
3. For each node in the snapshot:
   - Check if the node exists in production
   - If it exists, restore its properties and labels
   - If it doesn't exist (was created during merge), delete it
4. For each relationship in the snapshot:
   - Check if the relationship exists in production
   - If it exists, restore its properties
   - If it doesn't exist (was created during merge), delete it
5. Commit the transaction
6. Update the merge status to CANCELLED

### Error Handling

The rollback process includes comprehensive error handling:
- Transaction rollback if any step fails
- Detailed error logging
- Status updates in the merge progress tracker
- Error information in the rollback response

## API Usage

### Rollback Endpoint

```
POST /merge/{merge_id}/rollback
```

Request body:
```json
{
  "rollback_type": "COMPLETE",  // or "PARTIAL"
  "entity_ids": ["node1", "node2"]  // Required only for PARTIAL rollback
}
```

Response:
```json
{
  "rollback_id": "rollback_123456",
  "merge_id": "merge_123456",
  "status": "successful",
  "nodes_reverted": 10,
  "relationships_reverted": 5,
  "error": null
}
```

### Automatic Rollback in Validation

When using the validation service, you can enable automatic rollback:

```python
validation_result = await validation_service.validate_merge(
    merge_id="merge_123456",
    transform_id="transform_123456",
    auto_rollback=True  # Enable automatic rollback on validation failure
)
```

The validation result will include metadata about the rollback:

```json
{
  "valid": false,
  "issues": [...],
  "critical_count": 2,
  "warning_count": 1,
  "info_count": 0,
  "metadata": {
    "auto_rollback_performed": true,
    "rollback_id": "rollback_123456",
    "nodes_reverted": 10,
    "relationships_reverted": 5
  }
}
```

## Best Practices

1. **Snapshot TTL**: Configure an appropriate TTL for snapshots based on your system's needs. Longer TTLs provide more time for rollbacks but consume more memory.

2. **Validation First**: When possible, use validation before applying changes to prevent the need for rollbacks.

3. **Monitoring**: Monitor rollback operations and analyze patterns to identify recurring issues in merge operations.

4. **Testing**: Thoroughly test rollback functionality in non-production environments before relying on it in production.

5. **Partial Rollbacks**: Use partial rollbacks with caution, as they may leave the graph in an inconsistent state if dependencies between entities are not properly considered.

## Limitations

1. **Snapshot Size**: Very large merges affecting thousands of nodes may create large snapshots that consume significant memory.

2. **TTL Constraints**: Rollbacks can only be performed within the TTL period of the snapshot.

3. **External Systems**: Changes to external systems triggered by a merge are not automatically reverted.

4. **Concurrent Modifications**: If nodes are modified by other processes after a merge but before a rollback, those changes will be lost during rollback.

## Future Enhancements

1. **Tiered Storage**: Move snapshots to disk-based storage for longer retention with less memory impact.

2. **Selective Property Rollback**: Allow rollback of specific properties rather than entire entities.

3. **Rollback Preview**: Provide a preview of changes that would be made during a rollback.

4. **Cascading Rollbacks**: Intelligently handle dependencies between entities during partial rollbacks. 