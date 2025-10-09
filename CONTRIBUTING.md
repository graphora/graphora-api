# Contributing to Graphora API

Thank you for your interest in contributing to Graphora! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Contributor License Agreement](#contributor-license-agreement)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.11+
- uv (recommended) or pip
- Git
- Neo4j database (for local testing)
- LLM API access (OpenAI, Anthropic, or others)

### Ways to Contribute

- **Bug Reports**: File detailed bug reports with reproduction steps
- **Feature Requests**: Propose new features with clear use cases
- **Code Contributions**: Fix bugs, implement features, improve performance
- **Documentation**: Improve README, add examples, write tutorials
- **Testing**: Write tests, improve test coverage
- **Performance**: Optimize algorithms, reduce latency

## Development Setup

1. **Fork the repository**
   ```bash
   # Click the "Fork" button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/graphora-api.git
   cd graphora-api
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/graphora/graphora-api.git
   ```

4. **Install uv (recommended)**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

5. **Create virtual environment and install dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

6. **Install system dependencies**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install libmagic1

   # On macOS
   brew install libmagic
   ```

7. **Set up environment variables**
   ```bash
   cp .env.sample .env
   # Edit .env with your configuration
   ```

8. **Set up BAML**
   ```bash
   # Follow instructions at https://docs.boundaryml.com/guide/installation-language/python
   ```

9. **Run the server**
   ```bash
   python -m app.main
   ```

The API will be available at http://localhost:8000

## Making Changes

### Branch Naming Convention

Use descriptive branch names:
- `feature/your-feature-name` - For new features
- `fix/bug-description` - For bug fixes
- `docs/what-you-are-documenting` - For documentation
- `refactor/what-you-are-refactoring` - For code refactoring
- `test/what-you-are-testing` - For adding tests
- `perf/what-you-are-optimizing` - For performance improvements

Example:
```bash
git checkout -b feature/entity-extraction-improvements
```

### Commit Messages

Write clear, concise commit messages:
- Use present tense ("Add feature" not "Added feature")
- First line should be 50 characters or less
- Reference issues and PRs when applicable

Example:
```
Improve entity extraction accuracy

- Add context window expansion for better extraction
- Implement confidence scoring for entities
- Add unit tests for new extraction logic

Fixes #123
```

## Pull Request Process

1. **Ensure your code works**
   ```bash
   python -m app.main  # Test server starts
   ```

2. **Run linting**
   ```bash
   ruff check .
   black --check .
   ```

3. **Run tests**
   ```bash
   pytest
   ```

4. **Type checking (if using mypy)**
   ```bash
   mypy app/
   ```

5. **Update documentation**
   - Update README if you changed functionality
   - Add docstrings to new functions/classes
   - Update API documentation if endpoints changed

6. **Create Pull Request**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link related issues
   - Add examples for API changes
   - Mark as draft if work is in progress

7. **Sign the CLA**
   - All contributors must sign the [Contributor License Agreement](CLA.md)
   - Add this line to your PR description:
     ```
     I have read and agree to the Contributor License Agreement.
     ```

8. **Code Review**
   - Be responsive to feedback
   - Make requested changes promptly
   - Keep discussions respectful and constructive

## Coding Standards

### Python Style

- Follow PEP 8 style guide
- Use type hints for all functions
- Use Black for code formatting
- Use Ruff for linting
- Maximum line length: 100 characters

### Code Quality

- Write self-documenting code
- Use meaningful variable names
- Keep functions focused and small
- Avoid deep nesting

### Type Hints

Always use type hints:

```python
from typing import List, Optional, Dict

def process_documents(
    documents: List[str],
    ontology_id: str,
    options: Optional[Dict[str, Any]] = None
) -> List[Entity]:
    """
    Process documents and extract entities.

    Args:
        documents: List of document texts to process
        ontology_id: ID of the ontology to use
        options: Optional processing options

    Returns:
        List of extracted entities
    """
    # Implementation
```

### Project Structure

```
app/
├── agents/          # AI agents for workflows
├── api/            # FastAPI endpoints
│   ├── v1/        # API version 1
│   └── graphql/   # GraphQL endpoints
├── services/       # Business logic
├── schemas/        # Pydantic models
├── utils/          # Utility functions
└── main.py        # Application entry point
```

### Error Handling

- Use appropriate exception types
- Provide helpful error messages
- Log errors appropriately
- Return proper HTTP status codes

```python
from fastapi import HTTPException, status

def get_ontology(ontology_id: str) -> Ontology:
    ontology = db.query(Ontology).filter(Ontology.id == ontology_id).first()
    if not ontology:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ontology with id {ontology_id} not found"
        )
    return ontology
```

### Docstrings

Use Google-style docstrings:

```python
def extract_entities(text: str, ontology: Ontology) -> List[Entity]:
    """
    Extract entities from text using the provided ontology.

    Args:
        text: The input text to process
        ontology: The ontology defining entity types

    Returns:
        A list of extracted entities

    Raises:
        ValueError: If text is empty or ontology is invalid
        ExtractionError: If entity extraction fails

    Examples:
        >>> ontology = load_ontology("medical")
        >>> entities = extract_entities("Patient has diabetes", ontology)
        >>> print(entities[0].type)
        'Disease'
    """
```

### Async Code

- Use `async`/`await` for I/O operations
- Use `asyncio` for concurrent operations
- Properly handle async context managers

```python
async def process_document(document_id: str) -> ProcessingResult:
    async with get_db_session() as session:
        document = await session.get(Document, document_id)
        result = await extract_and_store(document)
        return result
```

## Testing

### Writing Tests

- Use pytest for all tests
- Write tests for new features
- Write tests for bug fixes
- Aim for >80% code coverage
- Test edge cases and error conditions

### Test Organization

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── fixtures/       # Test fixtures and data
```

### Test Example

```python
import pytest
from app.services.extraction import extract_entities

def test_entity_extraction_basic():
    """Test basic entity extraction functionality."""
    text = "John Smith works at Google"
    entities = extract_entities(text, ontology_id="test")

    assert len(entities) == 2
    assert entities[0].type == "Person"
    assert entities[0].value == "John Smith"
    assert entities[1].type == "Organization"
    assert entities[1].value == "Google"

@pytest.mark.asyncio
async def test_async_document_processing():
    """Test asynchronous document processing."""
    result = await process_document("test-doc-123")
    assert result.status == "completed"
    assert len(result.entities) > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/test_extraction.py

# Run tests matching pattern
pytest -k "test_entity"

# Run with verbose output
pytest -v
```

## Performance Considerations

- Profile code for performance bottlenecks
- Use batch processing for large datasets
- Implement caching where appropriate
- Monitor memory usage
- Use connection pooling for databases

## Database Migrations

If you make database schema changes:
1. Create a migration script
2. Test migration on a copy of production data
3. Document migration steps
4. Include rollback procedure

## API Documentation

- Document all endpoints with FastAPI docstrings
- Include request/response examples
- Document error responses
- Update OpenAPI schema

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: See [SECURITY.md](SECURITY.md)
- **General**: See [SUPPORT.md](SUPPORT.md)

## License

By contributing, you agree that your contributions will be licensed under the AGPL v3 License and that you have read and agreed to the [Contributor License Agreement](CLA.md).

---

**Thank you for contributing to Graphora!** 🎉
