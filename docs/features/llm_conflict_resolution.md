# LLM-Assisted Conflict Resolution

The GraphIT API now supports intelligent conflict resolution using Language Models (LLMs). This feature enhances the merge process by providing smart resolution options for various types of conflicts, along with confidence scores and explanations.

## Overview

During graph merges, conflicts can occur when there are discrepancies between source and target graphs. These conflicts need to be resolved before completing the merge. The LLM-assisted conflict resolution feature analyzes these conflicts using AI and suggests the most appropriate resolution options based on:

1. Conflict type and details
2. Ontology constraints and domain knowledge
3. Data patterns and standards

## Supported Conflict Types

The LLM analyzer supports the following conflict types:

### Property Value Conflicts

When the same entity has different property values in the source and target graphs.

**Example**: An entity representing a person has age=30 in the source graph and age=32 in the target graph.

**Resolution Options**:
- Use source value
- Use target value
- Use a standardized value derived from analysis

### Relationship Type Conflicts

When the same relationship has different types in the source and target graphs.

**Example**: A relationship between two entities is typed as "WORKS_FOR" in the source graph and "EMPLOYED_BY" in the target graph.

**Resolution Options**:
- Use source relationship type
- Use target relationship type
- Use a standardized relationship type based on ontology

### Duplicate Entity Conflicts

When entities in the source and target graphs are identified as potential duplicates.

**Example**: Two entities representing the same person with slight variations in properties.

**Resolution Options**:
- Merge the entities
- Keep entities separate
- Apply custom merge strategy

## How It Works

1. The system detects conflicts during the merge process
2. The LLM analyzer examines each conflict in context
3. The analyzer generates resolution options with confidence scores
4. Each option includes an explanation of the reasoning
5. Users can select the preferred option or add custom resolutions
6. The system applies the selected resolutions to complete the merge

## API Usage

### Analyze Conflicts with LLM

```http
POST /api/v1/merge/{merge_id}/conflicts/analyze
Content-Type: application/json

{
  "conflict_ids": ["conflict1", "conflict2"]  // Optional, if omitted all unresolved conflicts are analyzed
}
```

#### Response

```json
{
  "total_conflicts": 2,
  "analyzed": 2
}
```

### Get Conflicts with Analysis Results

```http
GET /api/v1/merge/{merge_id}/conflicts
```

#### Response

```json
{
  "conflicts": [
    {
      "id": "conflict1",
      "type": "PROPERTY_VALUE",
      "resolved": false,
      "details": {
        "entity_id": "entity1",
        "property_name": "age",
        "source_value": "30",
        "target_value": "32"
      },
      "resolution_options": [
        {
          "value": "32",
          "confidence": 0.8,
          "explanation": "The target value is more recent and likely reflects the current age."
        },
        {
          "value": "30",
          "confidence": 0.2,
          "explanation": "The source value appears to be outdated."
        }
      ]
    }
  ],
  "total": 1
}
```

## Configuration

LLM-assisted conflict resolution requires OpenAI API credentials, which can be configured in the environment:

```
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.2
OPENAI_MAX_TOKENS=1000
```

## Fallback Mechanism

If the LLM analysis fails for any reason, the system falls back to default resolution options based on the conflict type. This ensures the merge process can continue even if the AI analysis is unavailable.

## Performance Considerations

- LLM analysis adds processing time to the conflict resolution workflow
- For large merges with many conflicts, consider analyzing conflicts in batches
- Response times depend on the LLM service's performance and availability

## Future Enhancements

- Support for additional conflict types
- Improved confidence scoring based on historical resolution patterns
- Integration with domain-specific ontologies for more accurate recommendations
- Offline mode using pre-trained models for environments without internet access
