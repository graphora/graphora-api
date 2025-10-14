.PHONY: dev start test help

help:
	@echo "Available commands:"
	@echo "  make dev    - Start development server with auto-reload"
	@echo "                (takes ~5s to start, watches app/ directory only)"
	@echo "  make start  - Start production server without auto-reload"
	@echo "  make test   - Run tests"

dev:
	LOG_LEVEL=DEBUG uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app

start:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	uv run pytest
