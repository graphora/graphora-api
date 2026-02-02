# Local Development Guide

This guide explains how to bring up the Graphora API locally with its core dependencies so you can exercise the transformation flow end to end.

> **Need a one-command Docker stack?**
> Check [`LOCAL_DEV_DOCKER.md`](LOCAL_DEV_DOCKER.md) for the 5-minute quickstart powered by
> `make dev-up`. Continue below if you prefer running the API on your host.

## Prerequisites

- Docker and Docker Compose (v2) for running Neo4j and Redis
- Python 3.11 with [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- Access to a PostgreSQL database (the docker stack ships one; a Supabase DSN also works)
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
      - `DATABASE_URL` pointing at the Postgres instance you want the API to use
      - Optional: `DOCKER_DATABASE_URL` if Docker containers should hit a different DSN than your host
      - Optional: `POSTGRES_DATA_SOURCE` if you need Postgres data stored on a specific host path (defaults
        to the git-ignored `./.docker-data/postgres` directory next to this repo)
      - Optional: `NEO4J_STAGING_*_SOURCE` / `NEO4J_PROD_*_SOURCE` to relocate each Neo4j instance’s data,
        logs, and import folders (defaults live under `./.docker-data/neo4j-<env>/`)
      - Optional: `REDIS_DATA_SOURCE` to relocate the Redis append-only dir (`./.docker-data/redis` by default)
   - Optional: `SUPABASE_URL`, `SUPABASE_KEY` (only needed if you still proxy Supabase APIs elsewhere)
   - LLM credentials for the AI config tables (see below)
   - `PREFECT_API_URL` (and optionally `PREFECT_API_KEY`)

> **Tip:** keep `.env` outside of version control and rotate any temporary keys you use while testing.

## 3. Run local infrastructure

### Option A — all-in-one Docker stack (recommended)

```bash
make dev-up
```

This builds `docker/Dockerfile.dev`, launches Postgres with pgvector, runs every SQL file in
`migrations/` through the ephemeral `db-migrate` service, then starts Redis, Prefect, and the two
Neo4j instances before exposing the API at `PUBLIC_API_URL` (default `http://localhost:8000`). Use `make dev-down` to stop the stack
and remove containers/volumes, `make dev-logs` to tail the API logs, and `make dev-shell` to open a
bash session inside the API container. See [`LOCAL_DEV_DOCKER.md`](LOCAL_DEV_DOCKER.md) for the full
5-minute walkthrough.

### Option B — lightweight Compose (run API on host)

If you prefer to run the API on bare metal but still want containerized dependencies:

```bash
make compose-up   # or: docker compose -f docker-compose.local.yml up -d
```

This starts:
- **Postgres** (pgvector) on port 5433 (user: `graphora`, password: `graphora`, db: `graphora`)
- **Neo4j Staging** on ports 7475 (HTTP) and 7688 (Bolt)
- **Neo4j Production** on ports 8474 (HTTP) and 8687 (Bolt)
- **Redis** on port 6380
- **Prefect** on port 4200

Use `make compose-down` to stop services. Data persists in named Docker volumes so subsequent runs retain your data.

Other useful commands:
- `make compose-logs` – tail logs from all services
- `make compose-status` – show container status

### Neo4j connection strings

The Compose stack provisions Neo4j with username `neo4j` and password `test-password`. Use the following URIs when populating the `database_configs` table:

- Staging URI: `bolt://localhost:7687`
- Production URI: `bolt://localhost:7687` (change this to a dedicated instance for real deployments)

### Prefect

If you do not already run Prefect Cloud, start a local server in another terminal:

```bash
prefect server start
```

Update `.env` with the API URL emitted during startup (usually `http://127.0.0.1:4200/api`).

## 4. Apply application migrations

The API expects the Postgres database referenced by `DATABASE_URL` to contain the tables defined in
`migrations/`. Apply (or reapply) them with:

```bash
uv run python scripts/run_migrations.py
```

The script records each filename in `schema_migrations`, so subsequent runs only execute new files.
Point `DATABASE_URL` at your Supabase Postgres connection string if you prefer to manage data there.
When you run `make dev-up`, the lightweight `db-migrate` service (built from
`docker/Dockerfile.migrate` and bundled with only `psycopg` + `sqlparse`) executes the same script
automatically against the bundled Postgres container. The API image now keeps only system packages;
the entrypoint checks whether `/opt/graphora/.venv` (mapped to the `uv-venv` Docker volume) exists
and runs `uv sync --frozen` when needed. Use `FORCE_UV_SYNC=1 make dev-up` to refresh dependencies.

Schema overview:

| Table | Purpose |
| ----- | ------- |
| `database_configs` | Stores per-user staging & production Neo4j credentials (encrypted) |
| `configs` | Links a user to their staging/prod database config IDs |
| `ai_providers`, `ai_models`, `ai_provider_configs`, `user_ai_configs` | Manages LLM provider settings |
| `audit_trail` | Records operations for observability |
| `document_usage`, `llm_usage` | Tracks usage metrics |

If you are pointing at an existing database (Supabase or self-hosted), make sure the
`ENCRYPTION_MASTER_KEY` in `.env` matches what was used to encrypt any stored credentials.

> **Low disk?** Either run the relevant reset target (`make dev-reset-postgres`, `make dev-reset-neo4j`,
> `make dev-reset-redis`) to delete the git-ignored directories under `./.docker-data`, or set the
> corresponding `*_SOURCE` env var(s) to another host path before running `make dev-down && make dev-up`.

### Auth Bypass Mode (Recommended for Local Development)

To skip Clerk authentication setup entirely during local development, enable auth bypass mode:

**Backend (.env):**
```bash
AUTH_BYPASS_ENABLED=true
AUTH_BYPASS_USER_ID=local-dev-user
AUTH_BYPASS_EMAIL=dev@localhost
```

**Frontend (.env.local):**
```bash
NEXT_PUBLIC_AUTH_BYPASS=true
AUTH_BYPASS_USER_ID=local-dev-user
AUTH_BYPASS_EMAIL=dev@localhost
```

When auth bypass is enabled:
- All API requests are authenticated as the bypass user (no Authorization header required)
- The frontend skips Clerk entirely (no sign-in page)
- Database entries should use `local-dev-user` as the user ID

> **Warning:** Never enable auth bypass in production. The backend will refuse to start if `AUTH_BYPASS_ENABLED=true` and `ENVIRONMENT=production`.

### Minimum configuration for testing

1. Insert a `database_configs` row pointing to the local Neo4j instance (use the Compose credentials).
2. Insert matching `configs` entry for your user ID (use `local-dev-user` if auth bypass is enabled, or your Clerk user ID otherwise).
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
make test       # Execute pytest suite (writes coverage.xml and terminal summary)
make test-unit  # Run unit tests only (skips integration)
make test-integration  # Run integration tests only
uv sync --group dev  # Install optional dev tooling (Vulture)
make deadcode  # Run dead-code scan with Vulture
make dev-reset-neo4j  # Remove local Neo4j data directories
make dev-reset-postgres  # Remove local Postgres data directory
make dev-reset-redis  # Remove local Redis data directory
```

## Troubleshooting

- **Authentication errors**: verify your Clerk JWKS URL and audience/issuer values. Tokens must be signed with the keys exposed by Clerk.
- **Database auth errors**: ensure `DATABASE_URL`/`DOCKER_DATABASE_URL` are correct and that your
  Supabase service role key has privileges if you're targeting Supabase directly.
- **Neo4j connection timeouts**: confirm the container is healthy via `docker ps` and the bolt port (7687) is reachable.
- **LLM extraction failures**: check the Gemini quota and that your AI provider config is active.

For further assistance, add detailed logs from the API (`LOG_LEVEL=DEBUG`) and Prefect to diagnose pipeline issues.
