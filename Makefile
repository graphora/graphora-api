.PHONY: dev start test help lint format deadcode

help:
	@echo "Available commands:"
	@echo "  make dev    - Start development server with auto-reload"
	@echo "                (takes ~5s to start, watches app/ directory only)"
	@echo "  make start  - Start production server without auto-reload"
	@echo "  make test   - Run tests"
	@echo "  make lint   - Run Ruff and Black checks"
	@echo "  make format - Format the codebase with Black"
	@echo "  make deadcode - Run the dead code scanner"

dev:
	LOG_LEVEL=DEBUG uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app

start:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run black --check .

format:
	uv run black .

deadcode:
	uv run python scripts/find_dead_code.py
