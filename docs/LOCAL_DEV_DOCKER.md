# Docker Dev Quickstart (≈5 min)

This flow spins up the API plus Redis, Prefect, and dual Neo4j instances with one command while
letting you live-edit the code from your host machine.

## 1. Prereqs
- Docker Desktop (or any Docker Engine with Compose v2)
- `make` and `bash`
- Access credentials for Clerk, Supabase, LLM providers (these sit in `.env` as usual)

## 2. Bootstrap env vars
```bash
cp .env.example .env
```
Populate Supabase, Clerk, and provider keys. Docker Compose reads this file for Redis/Neo4j
container credentials, but the API itself still pulls Neo4j connection info from the Supabase
`database_configs` table. For local development, create staging/prod entries pointing to
`bolt://neo4j-staging:7687` and `bolt://neo4j-prod:7687` with the passwords from `.env`.

Key toggles:
- `API_PORT` / `PUBLIC_API_URL` — host port for FastAPI (defaults to `8000`).
- `CORS_ORIGINS` — comma-separated FE origins (e.g. `http://localhost:3000`).
- `NEO4J_*_PASSWORD` — credentials used to bootstrap the local Neo4j containers (does not affect
  runtime connections stored in Supabase).
- `DATABASE_URL` vs `DOCKER_DATABASE_URL` — host processes read `DATABASE_URL`
  (defaults to `postgresql://graphora:graphora@localhost:5432/graphora` once the stack runs),
  while containers override it with `DOCKER_DATABASE_URL`. Set the latter if you want the dockerized
  API to hit a remote Supabase/Postgres instance instead of the bundled container.
- `POSTGRES_DATA_SOURCE` — defaults to `./.docker-data/postgres` (a git-ignored host directory that
  Docker creates automatically). Override it if you want to pin data to a different disk such as an
  external drive.
- `NEO4J_STAGING_*_SOURCE` / `NEO4J_PROD_*_SOURCE` — bind-mount paths for each Neo4j instance’s
  `/data`, `/logs`, and `/import` directories. They default to git-ignored folders under
  `./.docker-data/neo4j-<env>/...` but you can point them to any other disk if you need more space.
- `REDIS_DATA_SOURCE` — defaults to `./.docker-data/redis`; override if you need the AppendOnly file
  on a different disk.
- Need to expose Redis on the host? Add a `ports` section for `graphora-redis` in
  `docker-compose.override.yml`; it is not published by default to avoid collisions if you already
  have a local Redis listening on 6379.

## 3. Start the stack
```bash
make dev-up
```
This builds `docker/Dockerfile.dev`, mounts the repo to `/app` for hot reloads, starts Postgres,
applies every SQL file in `migrations/` via the lightweight `db-migrate` helper (based on
`docker/Dockerfile.migrate`, so it only ships `psycopg` + `sqlparse`), and waits for Redis, Prefect,
and both Neo4j instances (with APOC installed and import/export enabled). The API container no
longer bakes the Python `.venv` into the image; instead the entrypoint runs `uv sync --frozen` the
first time `/opt/graphora/.venv` (backed by the `uv-venv` volume) is empty. Subsequent restarts reuse
that cached environment, so fresh builds stay tiny while steady-state boots remain fast. Set
`FORCE_UV_SYNC=1` in your environment if you need to redownload dependencies.

Health checks:
```bash
curl http://localhost:${API_PORT:-8000}/health
open http://localhost:${PREFECT_PORT:-4200}   # Prefect UI/API
open http://localhost:${NEO4J_STAGING_HTTP_PORT:-7474}  # Staging Neo4j browser
open http://localhost:${NEO4J_PROD_HTTP_PORT:-8474}     # Production Neo4j browser
```

## 4. Iterate
- Edit code locally; uvicorn reloads automatically inside the container.
- Watch logs: `make dev-logs`
- Open a shell in the API container: `make dev-shell`
- Tear down everything (containers + volumes): `make dev-down`
- Need custom ports/credentials? Copy `docker/docker-compose.override.example.yml` to
  `docker-compose.override.yml` and tweak as needed — Compose automatically loads it.

## 5. Wire the frontend
Point the Graphora frontend (or any HTTP client) to `PUBLIC_API_URL` (default
`http://localhost:8000`). CORS inherits from `CORS_ORIGINS`, so add every local origin you need.
No frontend container is started for you; run it separately and hit the API via the exposed host
port.

## 6. Services & ports
| Service        | Host Port | Notes |
| -------------- | --------- | ----- |
| API (FastAPI)  | `${API_PORT:-8000}` | Hot-reload via bind mount |
| Postgres (app DB) | `${POSTGRES_PORT:-5432}` | Backed by `./.docker-data/postgres`; `db-migrate` seeds schemas on boot |
| Redis          | `${REDIS_PORT:-6379}` | Data stored in `./.docker-data/redis` (AOF enabled) |
| Prefect 3      | `${PREFECT_PORT:-4200}` | `PREFECT_API_URL` auto-set to `http://prefect:4200/api` inside the stack |
| Neo4j (staging) | `${NEO4J_STAGING_HTTP_PORT:-7474}` / `${NEO4J_STAGING_BOLT_PORT:-7687}` | Data/logs/import live under `./.docker-data/neo4j-staging` |
| Neo4j (prod) | `${NEO4J_PROD_HTTP_PORT:-8474}` / `${NEO4J_PROD_BOLT_PORT:-8687}` | Uses `./.docker-data/neo4j-prod` by default |
| uv-venv volume | n/a | Stores `/opt/graphora/.venv` so `uv sync` only runs when dependencies change |

Supabase remains external — point `SUPABASE_URL` / `SUPABASE_KEY` at your staging project or
another shared environment. If you change any of the `*_SOURCE` paths, make sure the directories
exist and are writable before running `make dev-up`. Use `make dev-reset-postgres` to wipe
`./.docker-data/postgres` or `make dev-reset-neo4j` to delete `./.docker-data/neo4j-*` and start fresh.

## 7. Troubleshooting
- `make dev-up` hangs: run `docker compose -f docker-compose.dev.yml ps` to see which health check
  failed (typically Neo4j credentials or Prefect port collisions).
- API can’t reach Neo4j: confirm your Supabase `database_configs` entries point to
  `bolt://neo4j-staging:7687` / `bolt://neo4j-prod:7687` and that the passwords match the ones in
  `.env`.
- Need to re-run migrations manually? Execute `uv run python scripts/run_migrations.py` (with
  `DATABASE_URL` pointing at the target database) or `docker compose -f docker-compose.dev.yml run --rm db-migrate`.
- Frontend CORS errors: ensure `CORS_ORIGINS` lists every origin (comma-separated) and restart with
  `make dev-up` after editing `.env`.
- Postgres volume ballooned? Run `make dev-reset-postgres` (after `make dev-down`) to delete
  `./.docker-data/postgres` and recreate it on the next `make dev-up`.
- Neo4j out of space? `make dev-reset-neo4j` removes `./.docker-data/neo4j-*` and the old Docker
  volumes so the next `make dev-up` starts clean.
- Redis append-only dir corrupted? Run `make dev-reset-redis` (after `make dev-down`).
