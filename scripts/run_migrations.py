"""Apply SQL migrations in order, skipping ones already recorded."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import psycopg
import sqlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIGRATIONS_DIR = ROOT / "migrations"
SCHEMA_TABLE = "schema_migrations"


def _log(message: str) -> None:
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def _build_database_url() -> str:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn

    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD", "")
    port = os.getenv("POSTGRES_PORT", "5432")

    if host and database and user:
        auth = f":{password}" if password else ""
        return f"postgresql://{user}{auth}@{host}:{port}/{database}"

    raise RuntimeError(
        "DATABASE_URL or POSTGRES_* variables must be set before running migrations"
    )


def _ensure_migrations_table(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_TABLE} (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(f"SELECT filename FROM {SCHEMA_TABLE}")
        return {row[0] for row in cur.fetchall()}


def _ensure_extensions(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        for extension in ("uuid-ossp", "pgcrypto"):
            cur.execute(f'CREATE EXTENSION IF NOT EXISTS "{extension}";')


def _apply_migration(conn: psycopg.Connection, migration_file: Path) -> None:
    sql_text = migration_file.read_text()
    statements = [stmt.strip() for stmt in sqlparse.split(sql_text) if stmt.strip()]

    _log(f"Applying {migration_file.name} ({len(statements)} statements)")
    with conn.transaction():
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)
            cur.execute(
                f"INSERT INTO {SCHEMA_TABLE} (filename, applied_at) VALUES (%s, NOW())",
                (migration_file.name,),
            )


def main() -> None:
    dsn = _build_database_url()
    _log(f"Connecting to {dsn}")
    migrations = sorted(MIGRATIONS_DIR.glob("*.sql"), key=lambda path: path.name)
    if not migrations:
        _log("No migrations found; exiting")
        return

    with psycopg.connect(dsn) as conn:
        _ensure_extensions(conn)
        applied = _ensure_migrations_table(conn)

        for path in migrations:
            if path.name in applied:
                _log(f"Skipping {path.name} (already applied)")
                continue
            _apply_migration(conn, path)

    _log("Migrations complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - surfaced to calling process
        _log(f"Migration failed: {exc}")
        raise
