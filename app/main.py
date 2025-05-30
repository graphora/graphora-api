from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ontology import router as ontology_router
from app.api.transform import router as transform_router
from app.api.graph import router as graph_router
from app.api.merge import router as merge_router
from app.api.config import router as config_router
from app.api.audit import router as audit_router
from app.domain.healthcare import router as healthcare_router
from app.config import settings
from app.utils.logger import logger
from app.services.transform.prefect_client import configure_prefect

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Graphora API")
    # Add any additional startup tasks here
    configure_prefect()
    
    yield  # Server is running
    # Shutdown
    logger.info("Shutting down Graphora API")
    # Add any cleanup tasks here

# Initialize FastAPI app
app = FastAPI(
    title="Graphora API",
    version="1.0.0",
    docs_url=f"{settings.API_V1_STR}/docs",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Configure CORS
origins = ["*"]  # In production, you should specify actual origins
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
app.include_router(audit_router)
app.include_router(healthcare_router)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
