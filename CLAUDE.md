# GraphIt API Development Guide

## Commands
- Run all tests: `pytest -v`
- Run specific test file: `pytest tests/path/to/test_file.py -v`
- Run specific test: `pytest tests/path/to/test_file.py::test_name -v`
- Run unit tests: `pytest tests/unit/ -v`
- Run integration tests: `export INTEGRATION_TESTS=1 && pytest tests/integration/ -v`
- Run E2E tests: `export E2E_TESTS=1 && pytest tests/integration/test_resolution_history_e2e.py -v`

## Code Style
- Classes: CamelCase (e.g., `MergeService`)
- Functions/variables: snake_case (e.g., `get_conflict_resolution`)
- Use Pydantic models for data validation/serialization
- Async/await patterns for asynchronous code
- Type hints for function parameters and return values
- FastAPI for API endpoints
- Comprehensive exception handling with custom exceptions
- Structured logging throughout the application
- Well-documented function docstrings
- Tests for all new functionality

## Architecture
- Neo4j graph database + vector stores (Qdrant)
- LLM integration through BAML models
- Prefect for workflow orchestration
- Service layer pattern with dependency injection