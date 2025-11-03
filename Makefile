.PHONY: dev start test test-unit test-integration help lint format deadcode openapi-snapshot

help:
	@echo "Available commands:"
	@echo "  make dev    - Start development server with auto-reload"
	@echo "                (takes ~5s to start, watches app/ directory only)"
	@echo "  make start  - Start production server without auto-reload"
	@echo "  make test   - Run tests"
	@echo "  make test-unit - Run unit tests (excludes integration marked tests)"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint   - Run Ruff and Black checks"
	@echo "  make format - Format the codebase with Black"
	@echo "  make deadcode - Run the dead code scanner"
	@echo "  make openapi-snapshot - Generate OpenAPI schema snapshot"

dev:
	LOG_LEVEL=DEBUG uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app

start:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	uv run pytest

test-unit:
	uv run pytest -m "not integration"

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check .
	uv run black --check .

format:
	uv run black .

deadcode:
	PYTHONPATH=. uv run python scripts/find_dead_code.py

openapi-snapshot:
	PYTHONPATH=. uv run python scripts/openapi_snapshot.py
