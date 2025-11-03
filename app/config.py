from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables with validation."""

    # Test mode flag
    test_mode: bool = Field(
        default=False, description="Whether the application is running in test mode"
    )

    # API Settings
    API_V1_STR: str = Field(default="/api/v1", description="API version prefix")

    # Upload Settings
    UPLOAD_DIR: str = Field(
        default="/tmp/graphora/uploads",
        description="Directory for storing uploaded files",
    )

    ONTOLOGY_DIR: str = Field(
        default="/tmp/graphora/ontologies",
        description="Directory for storing ontologies",
    )

    # Storage Settings
    STORAGE_BATCH_SIZE: int = Field(
        default=1000, description="Batch size for storage operations"
    )
    STORAGE_RETRIES: int = Field(
        default=3, description="Number of retries for storage task"
    )

    # PDF Processor Settings
    PDF_PROCESSOR: str = Field(default="gemini", description="PDF processor to use")

    # Marker API Settings
    MARKER_API_HOST: str = Field(
        default="http://localhost:8000", description="Marker API host URL"
    )
    MARKER_API_TIMEOUT: int = Field(
        default=270,  # 4.5 minutes to allow for some buffer before task timeout
        description="Marker API request timeout in seconds",
    )
    MARKER_API_MAX_RETRIES: int = Field(
        default=3, description="Maximum retry attempts for Marker API"
    )
    MARKER_API_BACKOFF_FACTOR: float = Field(
        default=0.5, description="Exponential backoff factor for retries"
    )
    MARKER_API_USE_LLM: bool = Field(
        default=False, description="Whether to use LLM for PDF conversion"
    )
    MARKER_API_PAGINATE: bool = Field(
        default=True, description="Whether to paginate markdown output"
    )

    # Chunking Settings
    MAX_CHUNK_SIZE: int = Field(
        default=32000, description="Maximum size of a text chunk"
    )
    MIN_CHUNK_SIZE: int = Field(
        default=1000, description="Minimum size of a text chunk"
    )
    MIN_SEMANTIC_CHUNK_SIZE: int = Field(
        default=3000, description="Minimum size of a semantic text chunk"
    )
    MAX_CHUNKS_PER_DOC: int = Field(
        default=100, description="Maximum number of chunks per document"
    )
    SEMANTIC_THRESHOLD: float = Field(
        default=0.7, description="Threshold for semantic similarity in chunking"
    )
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="HuggingFace model for text embeddings",
    )
    CHUNKING_RETRIES: int = Field(
        default=3, description="Number of retries for chunking task"
    )
    TRANSFORM_RETRIES: int = Field(
        default=3, description="Number of retries for transformation task"
    )
    RETRY_DELAY_SECONDS: int = Field(
        default=30, description="Delay between retries in seconds"
    )
    EXTRACTION_LARGE_DOCUMENT_THRESHOLD: int = Field(
        default=5, description="Threshold for large document for parallel processing"
    )
    EXTRACTION_CONCURRENCY: int = Field(
        default=5, description="Concurrency for extraction"
    )
    EXTRACTION_RETRIES: int = Field(
        default=3, description="Number of retries for extraction task"
    )
    CHUNK_BATCH_SIZE: int = Field(default=5, description="Batch size for chunking")
    CHUNKING_MAX_CONCURRENCY: int = Field(
        default=4, description="Maximum number of documents chunked in parallel"
    )
    LLM_CACHE_MAX_ENTRIES: int = Field(
        default=128,
        description="Maximum number of LLM responses cached per process",
    )

    # Deterministic processing
    DETERMINISTIC_MODE: bool = Field(
        default=True,
        description="Enable stable IDs and deterministic contexts during transforms",
    )

    ENTITY_CANONICALIZATION_ENABLED: bool = Field(
        default=True,
        description="Enable ontology-driven canonicalization for entity properties",
    )

    # Timing Settings
    TIMING_WINDOW_HOURS: int = Field(
        default=24, description="Hours of timing data to keep for estimation"
    )

    # Redis Cache Settings
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")
    CACHE_TTL_HOURS: int = Field(default=24, description="Cache TTL in hours")

    CONFLICT_DETECTION_WORKERS: int = Field(
        default=5, description="Workers for conflict detection"
    )
    CONFLICT_BATCH_TTL: int = Field(
        default=24, description="Conflict batch TTL in hours"
    )
    NUMERIC_COMPARISON_TOLERANCE: float = Field(
        default=1e-6,
        description="Tolerance for comparing numeric values in conflict detection",
    )

    # Prefect Settings
    PREFECT_API_URL: str = Field(
        default="http://127.0.0.1:4200/api", description="Prefect API URL"
    )
    PREFECT_API_KEY: Optional[str] = Field(
        default=None, description="Prefect API key for authentication"
    )
    PREFECT_WORKPOOL_TRANSFORM: str = Field(
        default="transform", description="Prefect workpool for document transformation"
    )
    PREFECT_WORKPOOL_MERGE: str = Field(
        default="merge", description="Prefect workpool for document merging"
    )

    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL from components"""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Application logging level")

    # Merge Logging Settings
    LOG_DIR: str = Field(
        default="/tmp/graphora/logs", description="Directory for storing log files"
    )
    LOG_TO_DATABASE: bool = Field(
        default=True, description="Whether to store logs in database"
    )
    LOG_ROTATION_DAYS: int = Field(
        default=30, description="Number of days to keep logs before rotation"
    )
    LOG_MAX_FILE_SIZE_MB: int = Field(
        default=10, description="Maximum log file size in MB before rotation"
    )
    LOG_BACKUP_COUNT: int = Field(
        default=5, description="Number of backup log files to keep"
    )

    QUALITY_MIN_SCORE: float = Field(
        default=85.0,
        description="Minimum overall score required to auto-approve a transform",
    )
    QUALITY_FAIL_SCORE: float = Field(
        default=70.0,
        description="Minimum score required to proceed; below this the transform fails",
    )
    QUALITY_FAIL_ON_VIOLATION: bool = Field(
        default=True,
        description="Fail the transform when violations are present and auto-approval is off",
    )

    MERGE_ID_CONFIDENCE_THRESHOLD: float = Field(
        default=0.8,
        description="Minimum confidence required to treat a production match as authoritative when reconciling IDs",
    )

    MERGE_NODE_BATCH_SIZE: int = Field(
        default=250,
        ge=1,
        description="Number of nodes to persist per batch when writing merged graphs",
    )
    MERGE_REL_BATCH_SIZE: int = Field(
        default=500,
        ge=1,
        description="Number of relationships to persist per batch when writing merged graphs",
    )

    SUPABASE_URL: str = Field(default="", description="Supabase URL")
    SUPABASE_KEY: str = Field(default="", description="Supabase key")

    # Security Settings
    ENCRYPTION_MASTER_KEY: Optional[str] = Field(
        default=None,
        description="Master encryption key for password encryption (base64 encoded)",
    )

    # Auth Settings (Clerk)
    CLERK_JWKS_URL: str = Field(
        default="", description="URL to Clerk JWKS for token verification"
    )
    CLERK_ISSUER: str = Field(
        default="", description="Expected issuer for Clerk tokens"
    )
    CLERK_AUDIENCE: str = Field(
        default="", description="Expected audience for Clerk tokens"
    )
    CLERK_API_KEY: Optional[str] = Field(
        default=None, description="Clerk backend API key for management operations"
    )

    @property
    def ontology_dir(self) -> str:
        """Get the ontology directory path based on mode"""
        if self.test_mode:
            return "/tmp/graphora-test/ontologies"
        return self.ONTOLOGY_DIR

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Create and cache application settings."""
    try:
        settings = Settings()
        print(f"Loaded settings from environment with LOG_LEVEL={settings.LOG_LEVEL}")
        return settings
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        raise


settings = get_settings()
