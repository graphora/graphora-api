"""Async Postgres client backed by psycopg connection pool."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Iterable, Optional

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from app.config import settings
from app.utils.logger import logger

_pool: Optional[AsyncConnectionPool] = None
_pool_lock = asyncio.Lock()

_sync_pool: Optional[ConnectionPool] = None
_sync_pool_lock = Lock()


async def _create_pool() -> AsyncConnectionPool:
    dsn = settings.DATABASE_URL or settings.resolved_database_url
    if not dsn:
        raise ValueError("DATABASE_URL must be configured")

    logger.info(
        "Creating Postgres connection pool (min=%s, max=%s)",
        settings.DB_POOL_MIN_SIZE,
        settings.DB_POOL_MAX_SIZE,
    )

    return AsyncConnectionPool(
        conninfo=dsn,
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
        kwargs={"row_factory": dict_row},
    )


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                _pool = await _create_pool()
    return _pool


def _create_sync_pool() -> ConnectionPool:
    dsn = settings.DATABASE_URL or settings.resolved_database_url
    if not dsn:
        raise ValueError("DATABASE_URL must be configured")

    logger.info(
        "Creating sync Postgres connection pool (min=%s, max=%s)",
        settings.DB_POOL_MIN_SIZE,
        settings.DB_POOL_MAX_SIZE,
    )

    return ConnectionPool(
        conninfo=dsn,
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
        kwargs={"row_factory": dict_row},
    )


def get_sync_pool() -> ConnectionPool:
    global _sync_pool
    if _sync_pool is None:
        with _sync_pool_lock:
            if _sync_pool is None:
                _sync_pool = _create_sync_pool()
    return _sync_pool


async def fetch(query: str, *args: Any) -> list[dict[str, Any]]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, args or None)
            rows = await cur.fetchall()
            return rows


async def fetchrow(query: str, *args: Any) -> Optional[dict[str, Any]]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, args or None)
            row = await cur.fetchone()
            return row


async def fetchval(query: str, *args: Any) -> Any:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, args or None)
            row = await cur.fetchone()
            if row is None:
                return None
            return next(iter(row.values())) if isinstance(row, dict) else row


async def execute(query: str, *args: Any) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, args or None)


async def executemany(query: str, params: Iterable[Iterable[Any]]) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.executemany(query, params)


@asynccontextmanager
async def transaction():
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            async with conn.cursor() as cur:
                yield cur


def sync_fetch(query: str, *args: Any) -> list[dict[str, Any]]:
    pool = get_sync_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, args or None)
            return cur.fetchall()


def sync_fetchrow(query: str, *args: Any) -> Optional[dict[str, Any]]:
    pool = get_sync_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, args or None)
            return cur.fetchone()


def sync_fetchval(query: str, *args: Any) -> Any:
    row = sync_fetchrow(query, *args)
    if row is None:
        return None
    return next(iter(row.values())) if isinstance(row, dict) else row


def sync_execute(query: str, *args: Any) -> None:
    pool = get_sync_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, args or None)
