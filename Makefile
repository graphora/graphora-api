.PHONY: dev start test test-unit test-integration help lint format deadcode openapi-snapshot typecheck pre-commit dev-up dev-down dev-logs dev-shell dev-rebuild dev-reset-postgres dev-reset-neo4j dev-reset-redis

DEV_COMPOSE ?= docker compose -f docker-compose.dev.yml

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
	@echo "  make dev-up   - Build + start dockerized dev stack"
	@echo "  make dev-down - Stop stack, remove containers + volumes"
	@echo "  make dev-logs - Tail API logs from docker stack"
	@echo "  make dev-shell - Open a shell inside the API container"
	@echo "  make dev-reset-postgres - Delete the local Postgres data dir (./.docker-data/postgres)"
	@echo "  make dev-reset-neo4j - Delete the local Neo4j data dirs (./.docker-data/neo4j-*)"
	@echo "  make dev-reset-redis - Delete the local Redis data dir (./.docker-data/redis)"

dev:
	LOG_LEVEL=DEBUG uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app

start:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

dev-up:
	mkdir -p .docker-data/postgres \
		.docker-data/redis \
		.docker-data/neo4j-staging/data \
		.docker-data/neo4j-staging/logs \
		.docker-data/neo4j-staging/import \
		.docker-data/neo4j-prod/data \
		.docker-data/neo4j-prod/logs \
		.docker-data/neo4j-prod/import
	$(DEV_COMPOSE) up -d --build

dev-down:
	$(DEV_COMPOSE) down -v --remove-orphans

dev-logs:
	$(DEV_COMPOSE) logs -f api

dev-shell:
	$(DEV_COMPOSE) exec api bash

dev-rebuild:
	$(DEV_COMPOSE) build --pull --no-cache
	$(DEV_COMPOSE) up -d

dev-reset-postgres:
	rm -rf .docker-data/postgres
	-docker volume rm graphora-dev_postgres-data >/dev/null 2>&1 || true

dev-reset-neo4j:
	rm -rf .docker-data/neo4j-staging .docker-data/neo4j-prod
	-docker volume rm graphora-dev_neo4j-staging-data graphora-dev_neo4j-staging-logs graphora-dev_neo4j-staging-import \
		graphora-dev_neo4j-prod-data graphora-dev_neo4j-prod-logs graphora-dev_neo4j-prod-import >/dev/null 2>&1 || true

dev-reset-redis:
	rm -rf .docker-data/redis
	-docker volume rm graphora-dev_redis-data >/dev/null 2>&1 || true

test:
	uv run pytest

test-unit:
	uv run pytest -m "not integration"

test-integration:
	uv run pytest -m integration

lint-fix:
	uv run ruff check --fix .
	uv run black .

lint:
	uv run ruff check .
	uv run black --check .

format:
	uv run black .

deadcode:
	PYTHONPATH=. uv run python scripts/find_dead_code.py

openapi-snapshot:
	PYTHONPATH=. uv run python scripts/openapi_snapshot.py

typecheck:
	uv run mypy app

pre-commit:
	$(MAKE) lint-fix
# 	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) deadcode
