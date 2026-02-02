# Repository Guidelines

## Project Structure & Module Organization
- `app/main.py` boots the FastAPI service and registers routers declared under `app/api/*`.
- Place HTTP handlers in `app/api/<feature>` packages, keeping shared request/response models in `app/schemas/`.
- Orchestrate business logic inside `app/services/` (graph, transform, quality); cross-cutting helpers belong in `app/utils/`.
- Prefect tasks, ingestion flows, and background jobs sit in `app/services/transform/`; configuration defaults reside in `app/config.py` and `.env` files.
- Migration assets stay in `migrations/`, docs in `docs/`, and the sample system test in `test_quality_system.py` shows expected fixtures.

## Build, Test, and Development Commands
- `uv venv && uv sync` creates the Python 3.11 environment and installs dependencies.
- `make install` syncs dependencies with uv; `make install-dev` includes dev tools.
- `make compose-up` starts local Postgres, Neo4j, Prefect, and Redis (see `docs/LOCAL_DEVELOPMENT.md`).
- `make dev` starts the development server with auto-reload.
- `make test` executes the pytest suite with asyncio support and coverage.
- `make lint` runs Ruff + Black checks; `make lint-fix` applies auto-fixes; `make format` applies Black.
- `make typecheck` runs mypy after model or schema changes.
- `make migrate` runs database migrations against `DATABASE_URL`.
- `make help` shows all available make commands.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation and keep lines under 100 characters.
- Always type-hint functions; prefer dataclasses or Pydantic models over loose dicts.
- Use snake_case for modules and functions, PascalCase for classes, and SCREAMING_SNAKE_CASE for constants.
- Format code with Black and let Ruff handle lint/autofixes (`ruff --fix`) for small cleanups.

## Testing Guidelines
- Write `pytest` unit or async tests in a `tests/` package mirroring the module under test.
- Name files `test_<feature>.py` and functions `test_<behavior>_...` for automatic discovery.
- Use `pytest.mark.asyncio` for coroutine tests; rely on fixtures for Neo4j or external clients.
- Run focused checks with `uv run pytest tests/test_transform.py -k happy_path` when validating fixes.

## Commit & Pull Request Guidelines
- Create topic branches such as `feature/<summary>` or `fix/<issue>` and rebase regularly.
- Write imperative commit headers (≤50 chars) with contextual bodies referencing issues when relevant.
- Before opening a PR, run lint, formatting, type checks, and tests; attach outputs if failures remain.
- PR descriptions should summarize impact, list validation steps, link issues, and flag config or schema changes.
- Add screenshots or sample payloads for API changes, and confirm the CLA statement in the PR template.

## Environment & Secrets
- Copy `.env.example` to `.env` and populate Clerk (`CLERK_JWKS_URL`, `CLERK_ISSUER`, `CLERK_AUDIENCE`, `CLERK_API_KEY`), Supabase, and provider keys before running services.
- Install `libmagic` via Homebrew or apt for document inspection features, and follow BAML setup notes in `README.md`.
