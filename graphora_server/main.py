from contextlib import asynccontextmanager

import httpx
try:
    import redis.asyncio as redis_async
except ImportError:  # pragma: no cover — exercised without [redis] extra
    redis_async = None  # type: ignore
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from graphora_server.api.ai_config import router as ai_config_router
from graphora_server.api.audit import router as audit_router
from graphora_server.api.chat import router as chat_router
from graphora_server.api.chunking import router as chunking_router
from graphora_server.api.config import router as config_router
from graphora_server.api.dashboard import router as dashboard_router
from graphora_server.api.graph import router as graph_router
from graphora_server.api.merge import router as merge_router
from graphora_server.api.ontology import router as ontology_router
from graphora_server.api.quality import router as quality_router
from graphora_server.api.schema import router as schema_router
from graphora_server.api.transform import router as transform_router
from graphora_server.api.usage import router as usage_router
from graphora_server.config import settings
from graphora_server.services.transform.flows import progress_tracker
from graphora_server.services.transform.prefect_client import configure_prefect
from graphora_server.utils.logger import logger


def _create_limiter() -> Limiter:
    """Create rate limiter with Redis backend, falling back to memory for tests."""
    # Use in-memory storage for test mode to avoid Redis dependency
    if settings.test_mode:
        return Limiter(
            key_func=get_remote_address,
            default_limits=["100/minute"],
            storage_uri="memory://",
            strategy="fixed-window",
        )

    # Try Redis-backed storage for production (uses separate database to avoid conflicts)
    try:
        return Limiter(
            key_func=get_remote_address,
            default_limits=["100/minute"],
            storage_uri=settings.REDIS_RATE_LIMIT_URL,
            strategy="fixed-window",
        )
    except Exception as e:
        logger.warning(
            f"Failed to initialize Redis-backed rate limiter: {e}. "
            "Falling back to in-memory storage."
        )
        return Limiter(
            key_func=get_remote_address,
            default_limits=["100/minute"],
            storage_uri="memory://",
            strategy="fixed-window",
        )


limiter = _create_limiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("=" * 60, flush=True)
    print("GRAPHORA API STARTING...", flush=True)
    logger.info("Starting Graphora API")
    configure_prefect()

    # Clean up old transform directories on startup (older than 24 hours)
    try:
        progress_tracker.cleanup_old_transforms(max_age_hours=24)
        logger.info("Cleaned up old transform directories on startup")
    except Exception as e:
        logger.warning(f"Failed to cleanup old transforms on startup: {str(e)}")

    print("✓ GRAPHORA API READY - Server is now accepting requests", flush=True)
    print("=" * 60, flush=True)

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down Graphora API")


# Initialize FastAPI app
app = FastAPI(
    title="Graphora API",
    version="1.0.0",
    docs_url=f"{settings.API_V1_STR}/docs",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Add rate limiter to app state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS - require explicit configuration, warn if using wildcard
origins = settings.CORS_ORIGINS
if not origins:
    logger.warning(
        "CORS_ORIGINS not configured. Using wildcard '*' which allows any origin. "
        "This is not recommended for production."
    )
    origins = ["*"]
elif origins == ["*"]:
    logger.warning(
        "CORS_ORIGINS is set to wildcard '*'. "
        "This allows any origin and is not recommended for production."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes
app.include_router(ontology_router)
app.include_router(transform_router)
app.include_router(graph_router)
app.include_router(merge_router)
app.include_router(config_router)
app.include_router(ai_config_router)
app.include_router(audit_router)
app.include_router(usage_router)
app.include_router(schema_router)
app.include_router(chat_router)
app.include_router(quality_router)
app.include_router(chunking_router)
app.include_router(dashboard_router)


@app.get("/health")
@limiter.exempt  # Health checks should not be rate limited
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/system-info")
@limiter.exempt  # System info should not be rate limited
async def system_info():
    """System information endpoint exposing configuration for frontend"""
    return {
        "storage_type": settings.STORAGE_TYPE.lower(),
        "auth_bypass_enabled": settings.AUTH_BYPASS_ENABLED,
        "version": "1.0.0",
    }


@app.get("/ready")
@limiter.exempt  # Readiness checks should not be rate limited
async def readiness_check():
    """Readiness endpoint that probes core dependencies."""

    checks = {}

    # Redis probe
    if redis_async is None:
        checks["redis"] = {"status": "skipped", "detail": "redis extra not installed"}
    else:
        try:
            redis_client = redis_async.from_url(
                settings.REDIS_URL, encoding="utf-8", decode_responses=True
            )
            await redis_client.ping()
            checks["redis"] = {"status": "ok"}
            await redis_client.close()
        except Exception as exc:  # pragma: no cover - best-effort probing
            checks["redis"] = {"status": "error", "detail": str(exc)}

    # Prefect API probe (best-effort)
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            health_url = settings.PREFECT_API_URL.rstrip("/") + "/health"
            response = await client.get(health_url)
        if response.status_code == 200:
            checks["prefect"] = {"status": "ok"}
        else:
            checks["prefect"] = {
                "status": "degraded",
                "detail": f"HTTP {response.status_code}",
            }
    except Exception as exc:  # pragma: no cover - depends on environment
        checks["prefect"] = {"status": "error", "detail": str(exc)}

    overall_status = (
        "ok"
        if all(check.get("status") == "ok" for check in checks.values())
        else "degraded"
    )

    return {"status": overall_status, "checks": checks}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server")
    uvicorn.run(
        "graphora_server.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
