from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class DatabaseConfigBase(BaseModel):
    """Shared fields for Neo4j configuration objects."""

    id: Optional[str] = None
    name: str = Field(..., description="Human-readable name for the database")
    uri: str = Field(..., description="Neo4j connection URI")
    username: str = Field(..., description="Database username")

    @field_validator("uri")
    def validate_uri(cls, v):
        """Validate Neo4j URI format"""
        valid_prefixes = ["neo4j://", "bolt://", "neo4j+s://", "bolt+s://"]
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                "URI must start with neo4j://, bolt://, neo4j+s://, or bolt+s://"
            )
        return v


class DatabaseConfig(DatabaseConfigBase):
    """Schema for Neo4j database configuration"""

    password: str = Field(..., description="Database password")


class DatabaseConfigUpdate(BaseModel):
    """Schema for updating Neo4j configuration; fields optional."""

    name: Optional[str] = None
    uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    @field_validator("uri")
    @classmethod
    def validate_optional_uri(cls, value):
        if value is None:
            return value
        return DatabaseConfigBase.validate_uri(value)


class UserConfig(BaseModel):
    """Schema for user configuration containing staging and production databases"""

    id: Optional[str] = None
    userId: str = Field(..., description="Authenticated user identifier")
    stagingDb: DatabaseConfig = Field(..., description="Staging database configuration")
    prodDb: DatabaseConfig = Field(..., description="Production database configuration")
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

    @field_validator("prodDb")
    def validate_different_databases(cls, v, info):
        """Ensure staging and production databases have different URIs"""
        staging_db = info.data.get("stagingDb")
        if staging_db and v.uri == staging_db.uri:
            raise ValueError("Staging and production database URIs must be different")
        return v


class ConfigRequest(BaseModel):
    """Schema for creating user configuration"""

    stagingDb: DatabaseConfig = Field(..., description="Staging database configuration")
    prodDb: DatabaseConfig = Field(..., description="Production database configuration")


class ConfigUpdateRequest(BaseModel):
    """Schema for updating user configuration (password optional)."""

    stagingDb: Optional[DatabaseConfigUpdate] = Field(
        default=None, description="Staging database configuration"
    )
    prodDb: Optional[DatabaseConfigUpdate] = Field(
        default=None, description="Production database configuration"
    )


class ConfigResponse(BaseModel):
    """Schema for configuration API responses"""

    success: bool = True
    config: Optional[UserConfig] = None
    message: Optional[str] = None
    error: Optional[str] = None


class ConnectionTestRequest(BaseModel):
    """Schema for testing database connections"""

    uri: str = Field(..., description="Neo4j connection URI")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")


class ConnectionTestResponse(BaseModel):
    """Schema for connection test responses"""

    success: bool
    message: str
    error: Optional[str] = None
