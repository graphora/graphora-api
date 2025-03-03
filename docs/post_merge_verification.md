# Post-Merge Verification

## Overview

The Post-Merge Verification feature provides a mechanism to verify the integrity of a merge operation after it has been completed. This ensures that all nodes and relationships were correctly merged into the production graph and that the resulting graph maintains its integrity.

## Key Components

### 1. Verification Models

The verification process uses several models to represent verification data:

- **SnapshotData**: Captures a point-in-time snapshot of nodes and relationships in the graph
- **VerificationCheckType**: Enum of different types of verification checks
- **VerificationCheck**: Results of a specific verification check
- **VerificationResult**: Overall results of the verification process

### 2. PostMergeVerifier Service

The `PostMergeVerifier` class is responsible for performing verification checks on a merged graph. It performs the following checks:

- **Node Count Check**: Ensures that all nodes from the transformation are present in production
- **Relationship Count Check**: Ensures that all relationships from the transformation are present in production
- **Property Value Check**: Ensures that property values match between staging and production
- **Orphaned Node Check**: Ensures that no nodes are orphaned (have no relationships)
- **Ontology Constraint Check**: Ensures that the graph adheres to ontology constraints

### 3. API Endpoint

The verification functionality is exposed through a REST API endpoint:

```
POST /api/merge/{merge_id}/verify?transform_id={transform_id}
```

This endpoint triggers the verification process for a specific merge operation and transformation.

## Verification Process

The verification process follows these steps:

1. Retrieve the production graph data for the specified transformation
2. Perform verification checks on the graph data
3. Record the results of each check
4. Update the merge progress to reflect the verification status
5. Return the verification results

## Integration with Merge Flow

The verification stage is added as a new stage in the merge process:

1. Extract
2. Analyze
3. Conflict Detection
4. Resolution
5. Merge
6. Apply Changes
7. **Verification** (new stage)

## Error Handling

The verification process includes robust error handling:

- If any verification check fails, the overall verification is marked as failed
- Detailed information about failed checks is provided in the verification result
- The merge progress is updated to reflect the verification status

## Usage

### Verifying a Merge

To verify a merge operation, make a POST request to the verification endpoint:

```bash
curl -X POST "http://localhost:8000/api/merge/{merge_id}/verify?transform_id={transform_id}"
```

### Response Format

The response includes detailed information about the verification results:

```json
{
  "merge_id": "merge-123",
  "transform_id": "transform-456",
  "success": true,
  "timestamp": "2023-06-15T10:30:00Z",
  "checks": [
    {
      "check_type": "node_count",
      "success": true,
      "message": "All nodes present in production",
      "details": {
        "expected_count": 10,
        "actual_count": 10
      },
      "affected_entities": []
    },
    {
      "check_type": "relationship_count",
      "success": true,
      "message": "All relationships present in production",
      "details": {
        "expected_count": 15,
        "actual_count": 15
      },
      "affected_entities": []
    }
    // Additional checks...
  ]
}
```

## Troubleshooting

If verification fails, check the following:

1. Review the verification results to identify which checks failed
2. Examine the affected entities to understand which nodes or relationships have issues
3. Check the merge logs for any errors during the merge process
4. Verify that all conflicts were properly resolved before the merge

## Implementation Details

### Storage Interface

The verification process relies on the storage interface to retrieve production graph data:

```python
async def get_production_graph_for_transform(
    self,
    transform_id: str
) -> TransformationResult:
    """Get all nodes and relationships from production that were affected by a transform"""
```

### Progress Tracking

The verification process updates the merge progress to reflect the verification status:

```python
await self.progress_tracker.update_merge_stage(merge_id, MergeStage.VERIFICATION)
await self.progress_tracker.update_stage_status(
    merge_id, 
    MergeStage.VERIFICATION, 
    "in_progress"
)
```

## Testing

The verification functionality includes comprehensive tests:

- **Unit Tests**: Test the `PostMergeVerifier` class in isolation
- **Integration Tests**: Test the verification process end-to-end

## Future Enhancements

Potential future enhancements to the verification process:

1. **Automated Remediation**: Automatically fix issues detected during verification
2. **Verification Reports**: Generate detailed reports of verification results
3. **Scheduled Verification**: Run verification on a schedule to ensure ongoing graph integrity
4. **Custom Verification Checks**: Allow users to define custom verification checks 