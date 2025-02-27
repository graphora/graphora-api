# Resolution History Tests

This directory contains tests for the Resolution History feature, which tracks conflict resolutions and provides suggestions based on past resolutions.

## Test Types

The tests are organized into several categories:

1. **Unit Tests**: Test individual components in isolation with mocked dependencies
   - `tests/unit/services/test_resolution_history.py`: Tests for the `ResolutionHistoryService`
   - `tests/unit/services/merge/test_merge_service_resolution.py`: Tests for the integration between `MergeService` and `ResolutionHistoryService`

2. **API Tests**: Test the API endpoints with mocked services
   - `tests/api/test_resolution_history_api.py`: Tests for the resolution history API endpoints

3. **Integration Tests**: Test components with real dependencies (Redis)
   - `tests/integration/test_resolution_history_integration.py`: Tests for the `ResolutionHistoryService` with a real Redis instance
   - `tests/integration/test_merge_resolution_history_integration.py`: Tests for the integration between `MergeService` and `ResolutionHistoryService` with a real Redis instance

4. **End-to-End Tests**: Test the complete workflow from API to storage
   - `tests/integration/test_resolution_history_e2e.py`: Tests for the complete resolution history workflow

## Running the Tests

### Unit Tests

Unit tests can be run without any special setup:

```bash
pytest tests/unit/services/test_resolution_history.py -v
pytest tests/unit/services/merge/test_merge_service_resolution.py -v
```

### API Tests

API tests can also be run without special setup:

```bash
pytest tests/api/test_resolution_history_api.py -v
```

### Integration Tests

Integration tests require Redis to be running. They use a dedicated test database (db=15) to avoid interfering with production data.

To run the integration tests:

```bash
# Set the environment variable to enable integration tests
export INTEGRATION_TESTS=1

# Run the tests
pytest tests/integration/test_resolution_history_integration.py -v
pytest tests/integration/test_merge_resolution_history_integration.py -v
```

### End-to-End Tests

End-to-end tests also require Redis to be running:

```bash
# Set the environment variable to enable E2E tests
export E2E_TESTS=1

# Run the tests
pytest tests/integration/test_resolution_history_e2e.py -v
```

## Test Configuration

The tests use the following configuration:

- Redis URL is taken from `app.config.settings.REDIS_URL`
- Integration tests use database 15 (`db=15`) to avoid interfering with production data
- Tests clean up after themselves by calling `flushdb()` on the test database

## Adding New Tests

When adding new tests:

1. For unit tests, use mocks for all dependencies
2. For integration tests, use the `resolution_history_service` fixture which provides a real service with a test Redis database
3. For E2E tests, use the `test_client` fixture to make HTTP requests to the API

## Test Coverage

The tests cover:

1. Storing resolutions in the history
2. Retrieving resolution history
3. Finding similar resolutions for suggestions
4. Updating resolution success and feedback
5. Getting resolution statistics
6. Integration between MergeService and ResolutionHistoryService
7. API endpoints for resolution history

## Troubleshooting

If integration tests fail:

1. Ensure Redis is running
2. Check that the Redis URL in `app.config.settings` is correct
3. Verify that database 15 is available for testing
4. Make sure the environment variables are set correctly 