# BAML Usage Tracking Integration

This document describes the BAML (BoundaryML) usage tracking integration for cost monitoring and pricing analytics.

## Overview

The BAML usage tracking system automatically captures token usage and costs for all BAML function calls throughout the application. It integrates with the existing usage tracking system to provide comprehensive cost analytics.

## Implementation

### Core Components

1. **BAML Usage Tracker** (`app/utils/baml_usage_tracker.py`)
   - `BAMLUsageTracker` class wraps BAML Collector
   - Extracts usage metrics from BAML function logs
   - Maps BAML providers to our ModelProvider enum
   - Integrates with usage tracking service

2. **LLM Client Integration** (`app/services/llm/client.py`)
   - All BAML function calls now accept `user_id`, `transform_id`, `document_usage_id`
   - Automatic usage tracking when user_id is provided
   - Backward compatible - works without tracking parameters

3. **Transform Integration** (`app/services/transform/`)
   - Graph transformer passes user_id to LLM calls
   - Transform tasks include usage tracking context
   - Document processing automatically tracked

## BAML Functions Tracked

All BAML functions are instrumented with usage tracking:

- `ExtractNodesFromChunk` - Entity extraction from text chunks
- `ExtractRelationshipsFromChunk` - Relationship extraction from text chunks  
- `InferRelationship` - Relationship inference between entities
- `StandardiseProperties` - Property standardization
- `ResolveEntities` - Entity resolution and deduplication

## Usage Examples

### Manual Tracking

```python
from app.utils.baml_usage_tracker import track_baml_function
from app.baml_client import b

# Track a BAML function call
result = await track_baml_function(
    user_id="user123",
    function_name="ExtractNodesFromChunk", 
    operation_type="entity_extraction",
    baml_function_call=b.ExtractNodesFromChunk,
    chunk_text,
    context,
    {"tb": type_builder},
    transform_id="transform456",
    document_usage_id="doc789"
)
```

### Automatic Tracking (Recommended)

```python
from app.services.llm.client import LLMClient

llm_client = LLMClient()

# Usage tracking happens automatically when user_id is provided
result = await llm_client.extract_nodes_from_chunk(
    chunk="Sample text...",
    response_model=MyModel,
    user_id="user123",          # Enables tracking
    transform_id="transform456",
    document_usage_id="doc789"
)
```

## Data Tracked

For each BAML function call, the following data is captured:

- **Token Usage**: Input tokens, output tokens, total tokens
- **Cost Estimation**: Based on underlying model pricing
- **Performance**: Latency, success/failure status
- **Context**: Operation type, transform ID, document ID
- **Provider Info**: Underlying LLM provider and model used

## Database Schema

BAML usage is stored in the `llm_usage` table with:

- `model_provider`: Set to "baml"
- `model_name`: Extracted from BAML client configuration
- `operation_type`: Prefixed with "baml_" (e.g., "baml_ExtractNodesFromChunk")
- Token counts and cost estimates from Collector usage data

## Pricing Configuration

BAML functions use pricing based on their underlying providers:

```python
ModelProvider.BAML: {
    "unknown": {"input": 1.00, "output": 3.00},        # Fallback rates
    "gpt-4": {"input": 30.00, "output": 60.00},        # OpenAI rates
    "claude-3-sonnet": {"input": 3.00, "output": 15.00}, # Anthropic rates
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30} # Google rates
}
```

## API Integration

The tracking integrates with existing usage reporting APIs:

- `GET /api/v1/usage/report` - Includes BAML usage in reports
- `GET /api/v1/usage/models` - Shows breakdown by BAML functions
- `GET /api/v1/usage/limits` - Includes BAML costs in limit checks

## Error Handling

- **Graceful Fallback**: If tracking fails, BAML functions continue working
- **Logging**: Usage tracking errors are logged but don't interrupt processing
- **Retry Logic**: Failed tracking attempts are logged for investigation

## Configuration

### Required Environment Variables

```bash
# For usage tracking (same as existing system)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
ENCRYPTION_MASTER_KEY=your_encryption_key
```

### BAML Configuration

Ensure your BAML configuration includes appropriate clients. The tracker will automatically detect:

- Provider type (OpenAI, Anthropic, Google, etc.)
- Model names from client configuration
- Usage metadata from responses

## Testing

To test BAML usage tracking:

1. **Unit Tests**: Test collector integration and usage extraction
2. **Integration Tests**: Test end-to-end tracking in transform flows
3. **Manual Testing**: Use development environment with real BAML calls

Example test:

```python
from app.utils.baml_usage_tracker import BAMLUsageTracker

tracker = BAMLUsageTracker(
    user_id="test-user",
    operation_type="test_extraction"
)

# Use tracker.collector in BAML function calls
# Verify usage is captured in tracker.get_usage_summary()
```

## Monitoring and Analytics

Usage data can be analyzed for:

- **Cost Optimization**: Identify expensive operations
- **Performance Monitoring**: Track latency trends
- **Usage Patterns**: Understand function call frequency
- **Error Analysis**: Monitor failure rates by operation type

## Migration

If you have existing BAML usage without tracking:

1. The system automatically detects when tracking is available
2. Old functions continue working without modification
3. New usage tracking starts immediately when user_id is provided
4. No data migration required - tracking starts fresh

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `baml-py` is installed and up to date
2. **Missing Usage Data**: Check that user_id is passed to LLM client methods
3. **Cost Calculation Errors**: Verify model pricing configuration
4. **Database Errors**: Check Supabase connection and schema

### Debug Mode

Enable debug logging to see tracking details:

```python
import logging
logging.getLogger('app.utils.baml_usage_tracker').setLevel(logging.DEBUG)
```

## Future Enhancements

Potential improvements:

- **Real-time Monitoring**: Dashboard for live usage tracking
- **Cost Alerts**: Automated notifications for usage thresholds
- **Advanced Analytics**: ML-based cost optimization recommendations
- **A/B Testing**: Compare costs between different BAML configurations