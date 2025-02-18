from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables with validation."""
    
    # API Settings
    API_V1_STR: str = Field(
        default="/api/v1",
        description="API version prefix"
    )
    
    # Upload Settings
    UPLOAD_DIR: str = Field(
        default="/tmp/graphit/uploads",
        description="Directory for storing uploaded files"
    )
    
    ONTOLOGY_DIR:str = Field(
        default="/tmp/graphit/ontologies",
        description="Directory for storing ontologies"
    )
    
    # Neo4j Settings
    NEO4J_URI: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j database connection URI"
    )
    NEO4J_USER: str = Field(
        default="neo4j",
        description="Neo4j database username"
    )
    NEO4J_PASSWORD: str = Field(
        default="password",
        description="Neo4j database password"
    )
    NEO4J_DB: str = Field(
        default="neo4j",
        description="Neo4j database name"
    )
    
    # Storage Settings
    STORAGE_BATCH_SIZE: int = Field(
        default=1000,
        description="Batch size for storage operations"
    )
    STORAGE_RETRIES: int = Field(
        default=3,
        description="Number of retries for storage task"
    )
    
    # Staging Neo4j Settings
    STAGING_NEO4J_URI: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j staging database URI"
    )
    STAGING_NEO4J_USER: str = Field(
        default="neo4j",
        description="Neo4j staging database username"
    )
    STAGING_NEO4J_PASSWORD: str = Field(
        default="",
        description="Neo4j staging database password"
    )
    STAGING_NEO4J_DATABASE: str = Field(
        default="neo4j",
        description="Neo4j staging database name"
    )
    
    # Marker API Settings
    MARKER_API_HOST: str = Field(
        default="http://localhost:8000",
        description="Marker API host URL"
    )
    MARKER_API_TIMEOUT: int = Field(
        default=120,
        description="Marker API request timeout in seconds"
    )
    MARKER_API_MAX_RETRIES: int = Field(
        default=3,
        description="Maximum retry attempts for Marker API"
    )
    MARKER_API_BACKOFF_FACTOR: float = Field(
        default=0.5,
        description="Exponential backoff factor for retries"
    )
    MARKER_API_USE_LLM: bool = Field(
        default=False,
        description="Whether to use LLM for PDF conversion"
    )
    MARKER_API_PAGINATE: bool = Field(
        default=True,
        description="Whether to paginate markdown output"
    )

    # Chunking Settings
    MAX_CHUNK_SIZE: int = Field(
        default=1000,
        description="Maximum size of a text chunk"
    )
    MIN_CHUNK_SIZE: int = Field(
        default=100,
        description="Minimum size of a text chunk"
    )
    SEMANTIC_THRESHOLD: float = Field(
        default=0.7,
        description="Threshold for semantic similarity in chunking"
    )
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="HuggingFace model for text embeddings"
    )
    CHUNKING_RETRIES: int = Field(
        default=3,
        description="Number of retries for chunking task"
    )
    RETRY_DELAY_SECONDS: int = Field(
        default=30,
        description="Delay between retries in seconds"
    )
    EXTRACTION_RETRIES: int = Field(
        default=3,
        description="Number of retries for extraction task"
    )
    CHUNK_BATCH_SIZE: int = Field(
        default=5,
        description="Batch size for chunking"
    )

    # Redis Cache Settings
    REDIS_HOST: str = Field(
        default="localhost",
        description="Redis host"
    )
    REDIS_PORT: int = Field(
        default=6379,
        description="Redis port"
    )
    REDIS_DB: int = Field(
        default=0,
        description="Redis database number"
    )
    REDIS_PASSWORD: Optional[str] = Field(
        default=None,
        description="Redis password"
    )
    CACHE_TTL_HOURS: int = Field(
        default=24,
        description="Cache TTL in hours"
    )
    
    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL from components"""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # LLM Settings
    DEEPSEEK_API_KEY: Optional[str] = Field(
        default=None,
        description="DeepSeek API key for LLM operations"
    )
    DEEPSEEK_BASE_URL: Optional[str] = Field(
        default=None,
        description="DeepSeek API base URL for LLM operations"
    )
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM operations"
    )
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API key for LLM operations"
    )
    GOOGLE_GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google Gemini API key for LLM operations"
    )
    VERTEXAI_PROJECT_ID: Optional[str] = Field(
        default=None,
        description="Google VertexAI Project ID for LLM operations"
    )
    VERTEXAI_LOCATION: Optional[str] = Field(
        default='us-east5',
        description="Google VertexAI Project Location for LLM operations"
    )
    
    # GCP Settings
    GCP_PROJECT_ID: str = Field(
        default="",
        description="Google Cloud Project ID"
    )
    GCP_LOCATION: str = Field(
        default="us-central1",
        description="Google Cloud region"
    )
    
    MOCK_MODE: bool = Field(
        default=False,
        description="Enable mock mode for testing"
    )
    
    # Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Application logging level"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    """Create and cache application settings."""
    try:
        settings = Settings()
        if not settings.DEEPSEEK_API_KEY and not settings.OPENAI_API_KEY and not settings.ANTHROPIC_API_KEY and not settings.GOOGLE_GEMINI_API_KEY and not settings.VERTEXAI_PROJECT_ID:
            print("Warning: DEEPSEEK_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY/GOOGLE_GEMINI_API_KEY/VERTEXAI_PROJECT_ID not set")
        if not all([settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD, 
                    settings.STAGING_NEO4J_URI, settings.STAGING_NEO4J_USER, settings.STAGING_NEO4J_PASSWORD]):
            raise ValueError("Required Neo4j settings are missing")
        print(f"Loaded settings from environment with LOG_LEVEL={settings.LOG_LEVEL}")
        return settings
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        raise

settings = get_settings()
