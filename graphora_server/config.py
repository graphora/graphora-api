from functools import lru_cache
from typing import List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables with validation."""

    # Test mode flag
    test_mode: bool = Field(
        default=False,
        description="Whether the application is running in test mode",
        validation_alias="TEST_MODE",
    )

    # API Settings
    API_V1_STR: str = Field(default="/api/v1", description="API version prefix")
    API_PORT: int = Field(default=8000, description="Port for local API server")
    PUBLIC_API_URL: str = Field(
        default="http://localhost:8000", description="Base URL exposed to other apps"
    )
    CORS_ORIGINS: Union[List[str], str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        description="Comma-separated origins allowed for CORS",
    )

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
    STORAGE_TYPE: str = Field(
        default="neo4j",
        description="Graph storage type: 'neo4j' for production or 'memory' for local dev/demos",
    )
    STORAGE_BATCH_SIZE: int = Field(
        default=1000, description="Batch size for storage operations"
    )
    STORAGE_RETRIES: int = Field(
        default=3, description="Number of retries for storage task"
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
    CHUNKING_RANDOM_SEED: int = Field(
        default=42, description="Seed used to stabilise chunking heuristics"
    )
    SEMANTIC_MIN_DOC_LENGTH: int = Field(
        default=2000,
        description="Minimum document length (characters) before semantic chunking is enabled",
    )
    MAX_TXT_CHUNKS: int = Field(
        default=20, description="Maximum number of chunks allowed for text documents"
    )
    TRANSFORM_MAX_CONCURRENCY: int = Field(
        default=4,
        description="Maximum number of concurrent LLM extractions per transform",
    )
    MAX_CONTEXT_CHARS: int = Field(
        default=12000,
        description="Maximum number of characters to keep in LLM context prompts",
    )
    LLM_CACHE_MAX_ENTRIES: int = Field(
        default=128,
        description="Maximum number of LLM responses cached per process",
    )
    LLM_CACHE_URL: Optional[str] = Field(
        default=None,
        description="Optional Redis connection URL for sharing LLM response cache across workers",
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

    # Entity Resolution Settings
    ENTITY_RESOLUTION_EMBEDDING_ENABLED: bool = Field(
        default=True,
        description="Enable embedding-based semantic similarity for entity resolution",
    )
    ENTITY_RESOLUTION_EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for entity resolution embeddings",
    )
    ENTITY_RESOLUTION_SIMILARITY_THRESHOLD: float = Field(
        default=0.85,
        description="Minimum similarity threshold for entity matching",
    )
    ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED: bool = Field(
        default=True,
        description="Enable cross-document entity linking via entity store",
    )
    ENTITY_RESOLUTION_BATCH_SIZE: int = Field(
        default=500,
        description="Batch size threshold for entity resolution processing",
    )

    # Timing Settings
    TIMING_WINDOW_HOURS: int = Field(
        default=24, description="Hours of timing data to keep for estimation"
    )

    # Redis Cache Settings
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_RATE_LIMIT_DB: int = Field(
        default=1, description="Redis database number for rate limiting (isolated)"
    )
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

    @property
    def REDIS_RATE_LIMIT_URL(self) -> str:
        """Construct Redis URL for rate limiting (uses separate database)"""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_RATE_LIMIT_DB}"

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

    # Postgres / application database
    DATABASE_URL: str = Field(
        default="",
        description="SQLAlchemy/asyncpg compatible Postgres connection string",
    )
    POSTGRES_HOST: Optional[str] = Field(default=None)
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: Optional[str] = Field(default=None)
    POSTGRES_USER: Optional[str] = Field(default=None)
    POSTGRES_PASSWORD: Optional[str] = Field(default=None)
    DB_POOL_MIN_SIZE: int = Field(
        default=5,
        description="Minimum number of connections in the database pool",
    )
    DB_POOL_MAX_SIZE: int = Field(
        default=20,
        description="Maximum number of connections in the database pool",
    )

    # Security Settings
    ENCRYPTION_MASTER_KEY: Optional[str] = Field(
        default=None,
        description="Master encryption key for password encryption (base64 encoded)",
    )

    # Auth Settings (Clerk)
    AUTH_BYPASS_ENABLED: bool = Field(
        default=False,
        description="Enable auth bypass for local development (never use in production)",
    )
    AUTH_BYPASS_USER_ID: str = Field(
        default="local-dev-user",
        description="User ID to use when auth bypass is enabled",
    )
    AUTH_BYPASS_EMAIL: str = Field(
        default="dev@localhost",
        description="Email to use when auth bypass is enabled",
    )
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

    # LLM provider env-var fast path. When LLM_PROVIDER=ollama is set,
    # get_llm_client_for_user skips the DB lookup and serves the
    # client from these env values — the no-key local path. Production
    # multi-user deployments leave LLM_PROVIDER unset and configure
    # providers per-user via /api/v1/ai_config.
    LLM_PROVIDER: Optional[str] = Field(
        default=None,
        description=(
            "Override LLM provider via env. Currently 'ollama' is the only "
            "supported override; 'gemini' is the default DB-backed flow."
        ),
    )
    OLLAMA_HOST: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL when LLM_PROVIDER=ollama",
    )
    OLLAMA_MODEL: str = Field(
        default="llama3.2",
        description="Ollama model name when LLM_PROVIDER=ollama",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def split_cors_origins(cls, value):
        """Allow comma-separated env strings for CORS origins."""
        if isinstance(value, str):
            parts = [origin.strip() for origin in value.split(",") if origin.strip()]
            return parts or ["*"]
        return value

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def normalize_log_level(cls, value):
        """Normalize log level strings while allowing numeric overrides."""
        if isinstance(value, str):
            return value.upper()
        return value

    @property
    def ontology_dir(self) -> str:
        """Get the ontology directory path based on mode"""
        if self.test_mode:
            return "/tmp/graphora-test/ontologies"
        return self.ONTOLOGY_DIR

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def resolved_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        if self.POSTGRES_HOST and self.POSTGRES_DB and self.POSTGRES_USER:
            password = self.POSTGRES_PASSWORD or ""
            auth = f":{password}" if password else ""
            return (
                f"postgresql://{self.POSTGRES_USER}{auth}@{self.POSTGRES_HOST}:"
                f"{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        return ""


@lru_cache()
def get_settings() -> Settings:
    """Create and cache application settings."""
    try:
        settings = Settings()
        if not settings.DATABASE_URL:
            settings.DATABASE_URL = settings.resolved_database_url
        print(f"Loaded settings from environment with LOG_LEVEL={settings.LOG_LEVEL}")
        return settings
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        raise


settings = get_settings()
