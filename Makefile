.PHONY: install dev start test test-unit test-integration test-storage test-transform test-llm test-api test-api-unit test-audit test-chunking test-cache test-cov test-watch help lint format deadcode openapi-snapshot typecheck pre-commit compose-up compose-down compose-logs compose-status dev-up dev-down dev-logs dev-shell dev-rebuild dev-reset-postgres dev-reset-neo4j dev-reset-redis migrate clean

DEV_COMPOSE ?= docker compose -f docker-compose.dev.yml
LOCAL_COMPOSE ?= docker compose -f docker-compose.local.yml

help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install/sync Python dependencies via uv"
	@echo "  make install-dev      - Install with dev dependencies (Vulture, etc.)"
	@echo ""
	@echo "Development:"
	@echo "  make dev              - Start development server with auto-reload"
	@echo "  make start            - Start production server without auto-reload"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests (excludes integration marked tests)"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-cov         - Run tests with coverage report"
	@echo "  make test-watch       - Run tests in watch mode (requires pytest-watch)"
	@echo ""
	@echo "Component Tests:"
	@echo "  make test-storage     - Run storage layer tests"
	@echo "  make test-transform   - Run transform service tests"
	@echo "  make test-llm         - Run LLM client tests"
	@echo "  make test-api         - Run all API tests (unit + integration)"
	@echo "  make test-api-unit    - Run API unit tests only"
	@echo "  make test-audit       - Run audit service tests"
	@echo "  make test-chunking    - Run chunking config tests"
	@echo "  make test-cache       - Run cache infrastructure tests"
	@echo "  make test-entity-resolution - Run entity resolution tests"
	@echo "  make test-quality     - Run quality validation tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run Ruff and Black checks"
	@echo "  make lint-fix         - Run Ruff and Black with auto-fix"
	@echo "  make format           - Format the codebase with Black"
	@echo "  make deadcode         - Run the dead code scanner"
	@echo "  make typecheck        - Run mypy type checking"
	@echo "  make pre-commit       - Run all pre-commit checks"
	@echo ""
	@echo "Local Services (lightweight - run API on host):"
	@echo "  make compose-up       - Start Postgres, Neo4j, Prefect, and Redis"
	@echo "  make compose-down     - Stop local services"
	@echo "  make compose-logs     - Tail logs from local services"
	@echo "  make compose-status   - Show status of local services"
	@echo ""
	@echo "Docker Development (full stack in Docker):"
	@echo "  make dev-up           - Build + start dockerized dev stack"
	@echo "  make dev-down         - Stop stack, remove containers + volumes"
	@echo "  make dev-logs         - Tail API logs from docker stack"
	@echo "  make dev-shell        - Open a shell inside the API container"
	@echo "  make dev-rebuild      - Rebuild and restart docker stack"
	@echo ""
	@echo "Database & Migrations:"
	@echo "  make migrate          - Run database migrations"
	@echo "  make dev-reset-postgres - Delete local Postgres data"
	@echo "  make dev-reset-neo4j  - Delete local Neo4j data"
	@echo "  make dev-reset-redis  - Delete local Redis data"
	@echo ""
	@echo "Other:"
	@echo "  make openapi-snapshot - Generate OpenAPI schema snapshot"
	@echo "  make clean            - Remove build artifacts and cache files"

install:
	uv sync

install-dev:
	uv sync --group dev

dev:
	LOG_LEVEL=DEBUG uv run graphora-server serve --host 0.0.0.0 --port 8000 --reload

start:
	uv run graphora-server serve --host 0.0.0.0 --port 8000

compose-up:
	$(LOCAL_COMPOSE) up -d

compose-down:
	$(LOCAL_COMPOSE) down

compose-logs:
	$(LOCAL_COMPOSE) logs -f

compose-status:
	$(LOCAL_COMPOSE) ps

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

# Two env vars unblock the unit-suite stubs that would otherwise
# break integration tests:
#   GRAPHORA_TEST_REAL_NEO4J=1 — bypasses the Neo4j driver stub
#     (tests/conftest.py:_install_neo4j_stub).
#   GRAPHORA_TEST_REAL_DEPS=1  — bypasses the pandas/langchain/splink/
#     redis stubs (tests/conftest.py:_install_langchain_and_splink_stubs).
#     The real neo4j driver imports pandas at module load (pd.NA in
#     its packstream codec), so without the real pandas the driver
#     crashes on import. Setting it here so `make test-integration`
#     Just Works when Docker is up.
test-integration:
	GRAPHORA_TEST_REAL_NEO4J=1 GRAPHORA_TEST_REAL_DEPS=1 uv run pytest -m integration

test-cov:
	uv run pytest --cov=app --cov-report=html --cov-report=term-missing

test-watch:
	uv run pytest-watch -- -v

# Component-specific test targets
test-storage:
	uv run pytest tests/unit/services/storage/ -v

test-transform:
	uv run pytest tests/unit/services/transform/ -v

test-llm:
	uv run pytest tests/unit/services/llm/ -v

test-api-unit:
	uv run pytest tests/unit/api/ -v

test-api:
	uv run pytest tests/api/ tests/unit/api/ -v

test-quality:
	uv run pytest tests/quality/ -v

test-audit:
	uv run pytest tests/unit/services/audit/ -v

test-chunking:
	uv run pytest tests/unit/services/chunking/ -v

test-entity-resolution:
	uv run pytest tests/unit/services/entity_resolution/ -v

test-cache:
	uv run pytest tests/unit/services/cache/ -v

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

# Static check for the `WITH count(n) as ...` / `MATCH (n)`
# anti-pattern that silently scopes Cypher queries across the whole
# DB. The bug bit this codebase three times in two review rounds;
# this target is the preventive.
cypher-check:
	uv run python scripts/check_cypher_patterns.py

openapi-snapshot:
	PYTHONPATH=. uv run python scripts/openapi_snapshot.py

typecheck:
	uv run mypy graphora_server

pre-commit:
	$(MAKE) lint-fix
# 	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) deadcode

migrate:
	PYTHONPATH=. uv run python scripts/run_migrations.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ coverage.xml .coverage 2>/dev/null || true
