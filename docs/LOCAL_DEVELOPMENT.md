# Local Development Guide

This guide explains how to bring up the Graphora API locally with its core dependencies so you can exercise the transformation flow end to end.

## Prerequisites

- Docker and Docker Compose (v2) for running Neo4j and Redis
- Python 3.11 with [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- Access to a Supabase project (or compatible PostgREST + PostgreSQL instance)
- LLM provider credentials (Gemini via Google AI Platform is currently required)
- Prefect Cloud or a locally running Prefect server (v3+)

## 1. Clone and install dependencies

```bash
git clone https://github.com/graphora/graphora-api.git
cd graphora-api
uv sync
```

## 2. Prepare environment variables

1. Copy the template and populate real values:
   ```bash
   cp .env.example .env
   ```
2. Required keys:
   - `CLERK_JWKS_URL`, `CLERK_ISSUER`, `CLERK_AUDIENCE`, `CLERK_API_KEY`
   - `SUPABASE_URL`, `SUPABASE_KEY`
   - LLM credentials via Supabase AI config tables (see below)
   - `PREFECT_API_URL` (and optionally `PREFECT_API_KEY`)

> **Tip:** keep `.env` outside of version control and rotate any temporary keys you use while testing.

## 3. Run local infrastructure

The repository ships with a helper Compose file for Neo4j and Redis:

```bash
make compose-up  # starts neo4j:5.20 and redis:7 (ports 7474, 7687, 6379)
```

You can tear it down with `make compose-down`. Data persists in named Docker volumes so subsequent runs retain your test graph.

### Neo4j connection strings

The Compose stack provisions Neo4j with username `neo4j` and password `test-password`. Use the following URIs in your Supabase configuration:

- Staging URI: `bolt://localhost:7687`
- Production URI: `bolt://localhost:7687` (change this to a dedicated instance for real deployments)

### Prefect

If you do not already run Prefect Cloud, start a local server in another terminal:

```bash
prefect server start
```

Update `.env` with the API URL emitted during startup (usually `http://127.0.0.1:4200/api`).

## 4. Seed Supabase configuration tables

The API expects Supabase tables matching the following structure:

| Table | Purpose |
| ----- | ------- |
| `database_configs` | Stores per-user staging & production Neo4j credentials (encrypted) |
| `configs` | Links a user to their staging/prod database config IDs |
| `ai_providers`, `ai_models`, `ai_provider_configs`, `user_ai_configs` | Manages LLM provider settings |
| `audit_trail` | Records operations for observability |
| `document_usage`, `llm_usage` | Tracks usage metrics |

If you have not generated these tables yet, review your Supabase migration scripts or replicate the schema via the admin UI. Ensure the encryption master key in `.env` matches the one used for existing records.

### Minimum configuration for testing

1. Insert a `database_configs` row pointing to the local Neo4j instance (use the Compose credentials).
2. Insert matching `configs` entry for your Clerk user ID or email.
3. Add an AI provider config (Gemini) with a valid API key.

With those entries in place, the `/api/v1/config` and `/api/v1/ai-config` endpoints will succeed for your authenticated user.

## 5. Run the API

```bash
uv run python -m app.main
```

FastAPI will expose the service at `http://127.0.0.1:8000` by default. Documentation is available at `http://127.0.0.1:8000/api/v1/docs` and requires an authenticated Clerk session.

## 6. Execute an end-to-end transform

1. Authenticate via Clerk (obtain a bearer token or use the frontend application).
2. POST an ontology to `/api/v1/ontology`.
3. Upload documents via `/api/v1/transform/{ontology_id}/upload`.
4. Poll `/api/v1/transform/status/{transform_id}` to observe Prefect progress.
5. Query `/api/v1/graph/{transform_id}` to inspect the resulting nodes and relationships.

Refer to `docs/` for detailed API descriptions and the Python client quickstart for scripted workflows.

## 7. Useful helper commands

```bash
make lint       # Ruff + Black checks
make format     # Apply Black formatting
make typecheck  # Run mypy on the app package
make test       # Execute pytest suite
make test-unit  # Run unit tests only (skips integration)
make test-integration  # Run integration tests only
uv sync --group dev  # Install optional dev tooling (Vulture)
make deadcode  # Run dead-code scan with Vulture
```

## Troubleshooting

- **Authentication errors**: verify your Clerk JWKS URL and audience/issuer values. Tokens must be signed with the keys exposed by Clerk.
- **Supabase 401/404**: ensure the Supabase service role key is configured and tables are created.
- **Neo4j connection timeouts**: confirm the container is healthy via `docker ps` and the bolt port (7687) is reachable.
- **LLM extraction failures**: check the Gemini quota and that your AI provider config is active.

For further assistance, add detailed logs from the API (`LOG_LEVEL=DEBUG`) and Prefect to diagnose pipeline issues.
