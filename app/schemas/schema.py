from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class QuestionType(str, Enum):
    TEXT = "text"
    SELECT = "select"
    MULTISELECT = "multiselect"
    FILE = "file"
    TEXTAREA = "textarea"


class Question(BaseModel):
    id: str = Field(..., description="Unique question identifier")
    type: QuestionType = Field(..., description="Question type")
    prompt: str = Field(..., description="Question prompt text")
    required: bool = Field(default=True, description="Whether question is required")
    options: Optional[List[str]] = Field(default=None, description="Options for select/multiselect")
    placeholder: Optional[str] = Field(default=None, description="Placeholder text")
    help_text: Optional[str] = Field(default=None, description="Help text for the question")
    validation: Optional[Dict[str, Any]] = Field(default=None, description="Validation rules")


class QuestionSet(BaseModel):
    id: str = Field(..., description="Question set identifier")
    title: str = Field(..., description="Question set title")
    description: str = Field(..., description="Question set description")
    questions: List[Question] = Field(..., description="List of questions")
    conditions: Optional[List[str]] = Field(default=None, description="Conditions for showing this set")


class UserResponse(BaseModel):
    question_id: str = Field(..., description="Question identifier")
    value: Union[str, List[str]] = Field(..., description="User's answer")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class SchemaGenerationContext(BaseModel):
    domain: Optional[str] = Field(default=None, description="Domain category")
    use_case: Optional[str] = Field(default=None, description="Primary use case")
    data_types: Optional[List[str]] = Field(default=None, description="Types of data")
    complexity: Optional[str] = Field(default=None, description="Data complexity level")
    scale: Optional[str] = Field(default=None, description="Data scale/volume")
    temporal_requirements: Optional[str] = Field(default=None, description="Temporal tracking needs")


class SchemaGenerationRequest(BaseModel):
    user_responses: List[UserResponse] = Field(..., description="User's responses to questions")
    context: Optional[SchemaGenerationContext] = Field(default=None, description="Additional context")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Generation options")


class RelatedSchema(BaseModel):
    id: str = Field(..., description="Schema identifier")
    title: str = Field(..., description="Schema title")
    description: str = Field(..., description="Schema description")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    domain: str = Field(..., description="Schema domain")
    tags: List[str] = Field(default_factory=list, description="Schema tags")
    usage_count: int = Field(default=0, description="Number of times used")


class SchemaGenerationResponse(BaseModel):
    id: str = Field(..., description="Generated schema identifier")
    schema_content: str = Field(..., description="Generated YAML schema content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Generation confidence score")
    related_schemas: Optional[List[RelatedSchema]] = Field(default=None, description="Similar schemas found")
    suggestions: Optional[List[str]] = Field(default=None, description="Improvement suggestions")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Generation metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


# Schema Search Related Models
class SchemaSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    domain: Optional[str] = Field(default=None, description="Domain filter")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Similarity threshold")
    include_content: bool = Field(default=False, description="Include full schema content")


class SchemaSearchResult(BaseModel):
    id: str = Field(..., description="Schema identifier")
    title: str = Field(..., description="Schema title")
    description: str = Field(..., description="Schema description")
    content: Optional[str] = Field(default=None, description="Schema YAML content")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    domain: str = Field(..., description="Schema domain")
    tags: List[str] = Field(default_factory=list, description="Schema tags")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    usage_count: int = Field(default=0, description="Number of times used")
    user_id: str = Field(..., description="Creator user ID")


class SchemaSearchResponse(BaseModel):
    results: List[SchemaSearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original search query")
    took_ms: int = Field(..., description="Search execution time in milliseconds")


# Schema Storage Models
class StoredSchema(BaseModel):
    id: str = Field(..., description="Schema identifier")
    title: str = Field(..., description="Schema title")
    description: str = Field(..., description="Schema description")
    content: str = Field(..., description="YAML schema content")
    domain: str = Field(..., description="Schema domain")
    tags: List[str] = Field(default_factory=list, description="Schema tags")
    user_id: str = Field(..., description="Creator user ID")
    is_public: bool = Field(default=False, description="Whether schema is publicly available")
    usage_count: int = Field(default=0, description="Number of times used")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class CreateSchemaRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Schema title")
    description: str = Field(..., min_length=1, max_length=1000, description="Schema description") 
    content: str = Field(..., min_length=1, description="YAML schema content")
    domain: str = Field(..., description="Schema domain")
    tags: List[str] = Field(default_factory=list, description="Schema tags")
    is_public: bool = Field(default=False, description="Whether schema is publicly available")


class UpdateSchemaRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200, description="Schema title")
    description: Optional[str] = Field(default=None, min_length=1, max_length=1000, description="Schema description")
    content: Optional[str] = Field(default=None, min_length=1, description="YAML schema content") 
    domain: Optional[str] = Field(default=None, description="Schema domain")
    tags: Optional[List[str]] = Field(default=None, description="Schema tags")
    is_public: Optional[bool] = Field(default=None, description="Whether schema is publicly available")


# Question Configuration Models
class QuestionConfigRequest(BaseModel):
    domain: Optional[str] = Field(default=None, description="Domain to get questions for")
    include_optional: bool = Field(default=True, description="Include optional questions")


class QuestionConfigResponse(BaseModel):
    question_sets: List[QuestionSet] = Field(..., description="Available question sets")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Configuration metadata")


# Schema Refinement Models
class SchemaRefinementRequest(BaseModel):
    schema_id: str = Field(..., description="Schema to refine")
    user_feedback: str = Field(..., min_length=1, description="User's refinement request")
    current_schema: str = Field(..., description="Current schema content")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class SchemaRefinementResponse(BaseModel):
    refined_schema: str = Field(..., description="Refined schema content")
    changes_made: List[str] = Field(..., description="List of changes made")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Refinement confidence")
    explanation: str = Field(..., description="Explanation of changes")


# Usage Analytics Models
class SchemaUsageEvent(BaseModel):
    schema_id: str = Field(..., description="Schema that was used")
    user_id: str = Field(..., description="User who used the schema")
    event_type: str = Field(..., description="Type of usage event")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Event metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")


class SchemaAnalytics(BaseModel):
    schema_id: str = Field(..., description="Schema identifier")
    total_usage: int = Field(..., description="Total usage count")
    unique_users: int = Field(..., description="Number of unique users")
    recent_usage: List[SchemaUsageEvent] = Field(..., description="Recent usage events")
    popularity_score: float = Field(..., ge=0.0, description="Calculated popularity score")