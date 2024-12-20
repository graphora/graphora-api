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
    
    # LLM Settings
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
        default='asia-southeast1-b',
        description="Google VertexAI Project Location for LLM operations"
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
        if not settings.OPENAI_API_KEY and not settings.ANTHROPIC_API_KEY and not settings.GOOGLE_GEMINI_API_KEY and not settings.VERTEXAI_PROJECT_ID:
            print("Warning: OPENAI_API_KEY/ANTHROPIC_API_KEY/GOOGLE_GEMINI_API_KEY/VERTEXAI_PROJECT_ID not set")
        if not all([settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD]):
            raise ValueError("Required Neo4j settings are missing")
        print(f"Loaded settings from environment with LOG_LEVEL={settings.LOG_LEVEL}")
        return settings
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        raise

settings = get_settings()
