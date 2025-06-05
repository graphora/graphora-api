"""
Pydantic schemas for usage tracking and pricing.

This module defines the data models for tracking document processing
and LLM usage for pricing and analytics purposes.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal
from enum import Enum


class ProcessingStatus(str, Enum):
    """Status of document processing"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    IN_PROGRESS = "in_progress"


class PeriodType(str, Enum):
    """Time period types for aggregation"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ModelProvider(str, Enum):
    """LLM model providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BAML = "baml"


class DocumentUsage(BaseModel):
    """Schema for document processing usage tracking"""
    id: Optional[str] = None
    user_id: str = Field(..., description="User's ID")
    transform_id: str = Field(..., description="Transformation ID")
    
    # Document details
    document_name: str = Field(..., description="Name of the processed document")
    document_type: str = Field(..., description="File type (PDF, DOCX, etc.)")
    document_size_bytes: int = Field(..., description="File size in bytes")
    page_count: int = Field(default=0, description="Number of pages in document")
    
    # Processing details
    processing_status: ProcessingStatus = Field(..., description="Processing status")
    processing_started_at: datetime = Field(..., description="When processing started")
    processing_completed_at: Optional[datetime] = Field(None, description="When processing completed")
    processing_duration_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    
    # Quality metrics
    success_rate: Optional[Decimal] = Field(None, description="Success percentage")
    is_reprocessing: bool = Field(default=False, description="Whether this is a reprocessing")
    reprocessing_reason: Optional[str] = Field(None, description="Reason for reprocessing")
    
    # Extraction results
    chunks_created: int = Field(default=0, description="Number of chunks created")
    nodes_extracted: int = Field(default=0, description="Number of nodes extracted")
    relationships_extracted: int = Field(default=0, description="Number of relationships extracted")
    
    # Billing
    billable_pages: int = Field(default=0, description="Billable page count")
    billable_processing_units: int = Field(default=1, description="Processing units for billing")
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class LLMUsage(BaseModel):
    """Schema for LLM usage tracking"""
    id: Optional[str] = None
    user_id: str = Field(..., description="User's ID")
    transform_id: Optional[str] = Field(None, description="Associated transformation ID")
    document_usage_id: Optional[str] = Field(None, description="Associated document usage ID")
    
    # Model details
    model_provider: ModelProvider = Field(..., description="LLM provider")
    model_name: str = Field(..., description="Model name")
    model_version: Optional[str] = Field(None, description="Model version")
    
    # Token usage
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens generated")
    total_tokens: int = Field(default=0, description="Total tokens")
    
    # Cost tracking
    estimated_cost_usd: Optional[Decimal] = Field(None, description="Estimated cost in USD")
    cost_per_1k_input_tokens: Optional[Decimal] = Field(None, description="Input token cost rate")
    cost_per_1k_output_tokens: Optional[Decimal] = Field(None, description="Output token cost rate")
    
    # Operation details
    operation_type: str = Field(..., description="Type of operation")
    operation_context: Optional[str] = Field(None, description="Operation context")
    
    # Performance
    latency_ms: Optional[int] = Field(None, description="Response latency in milliseconds")
    success: bool = Field(default=True, description="Whether the operation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Timing
    request_timestamp: datetime = Field(..., description="When request was made")
    response_timestamp: Optional[datetime] = Field(None, description="When response received")
    
    created_at: Optional[datetime] = None
    
    @validator('total_tokens', always=True)
    def calculate_total_tokens(cls, v, values):
        """Calculate total tokens from input and output"""
        if v == 0:
            return values.get('input_tokens', 0) + values.get('output_tokens', 0)
        return v


class UsageAggregate(BaseModel):
    """Schema for aggregated usage statistics"""
    id: Optional[str] = None
    user_id: str = Field(..., description="User's ID")
    
    # Time period
    period_type: PeriodType = Field(..., description="Aggregation period type")
    period_start: datetime = Field(..., description="Period start time")
    period_end: datetime = Field(..., description="Period end time")
    
    # Document processing aggregates
    total_documents: int = Field(default=0, description="Total documents processed")
    total_pages: int = Field(default=0, description="Total pages processed")
    total_processing_time_ms: int = Field(default=0, description="Total processing time")
    avg_pages_per_document: Optional[Decimal] = Field(None, description="Average pages per document")
    success_rate: Optional[Decimal] = Field(None, description="Overall success rate")
    reprocessing_count: int = Field(default=0, description="Number of reprocessings")
    
    # Document type breakdown
    pdf_documents: int = Field(default=0, description="PDF documents processed")
    docx_documents: int = Field(default=0, description="DOCX documents processed")
    txt_documents: int = Field(default=0, description="TXT documents processed")
    other_documents: int = Field(default=0, description="Other document types")
    
    # LLM usage aggregates
    total_llm_calls: int = Field(default=0, description="Total LLM API calls")
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")
    total_estimated_cost_usd: Optional[Decimal] = Field(None, description="Total estimated cost")
    
    # Model breakdown (JSON for flexibility)
    model_usage_breakdown: Optional[Dict[str, Any]] = Field(None, description="Usage by model")
    
    # Performance metrics
    avg_tokens_per_page: Optional[Decimal] = Field(None, description="Average tokens per page")
    avg_processing_time_per_page_ms: Optional[int] = Field(None, description="Average processing time per page")
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PricingTier(BaseModel):
    """Schema for pricing tiers"""
    id: Optional[str] = None
    tier_name: str = Field(..., description="Name of the pricing tier")
    description: Optional[str] = Field(None, description="Tier description")
    
    # Limits
    monthly_document_limit: Optional[int] = Field(None, description="Monthly document limit")
    monthly_page_limit: Optional[int] = Field(None, description="Monthly page limit")
    monthly_token_limit: Optional[int] = Field(None, description="Monthly token limit")
    monthly_cost_limit_usd: Optional[Decimal] = Field(None, description="Monthly cost limit")
    
    # Features
    features: Optional[List[str]] = Field(None, description="Available features")
    
    # Pricing
    base_price_usd: Decimal = Field(default=Decimal('0'), description="Base monthly price")
    price_per_page_usd: Optional[Decimal] = Field(None, description="Price per page")
    price_per_1k_tokens_usd: Optional[Decimal] = Field(None, description="Price per 1K tokens")
    
    is_active: bool = Field(default=True, description="Whether tier is active")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UserPricingTier(BaseModel):
    """Schema for user tier assignments"""
    id: Optional[str] = None
    user_id: str = Field(..., description="User's ID")
    tier_id: str = Field(..., description="Assigned pricing tier ID")
    
    # Billing period
    billing_period_start: datetime = Field(..., description="Billing period start")
    billing_period_end: datetime = Field(..., description="Billing period end")
    
    # Current usage
    current_documents: int = Field(default=0, description="Current document count")
    current_pages: int = Field(default=0, description="Current page count")
    current_tokens: int = Field(default=0, description="Current token count")
    current_cost_usd: Decimal = Field(default=Decimal('0'), description="Current cost")
    
    # Status
    is_active: bool = Field(default=True, description="Whether assignment is active")
    over_limit: bool = Field(default=False, description="Whether user is over limits")
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DocumentUsageRequest(BaseModel):
    """Request schema for creating document usage records"""
    transform_id: str = Field(..., description="Transformation ID")
    document_name: str = Field(..., description="Document name")
    document_type: str = Field(..., description="Document type")
    document_size_bytes: int = Field(..., description="Document size")
    page_count: int = Field(default=0, description="Page count")


class LLMUsageRequest(BaseModel):
    """Request schema for creating LLM usage records"""
    transform_id: Optional[str] = Field(None, description="Transformation ID")
    document_usage_id: Optional[str] = Field(None, description="Document usage ID")
    model_provider: ModelProvider = Field(..., description="Model provider")
    model_name: str = Field(..., description="Model name")
    input_tokens: int = Field(..., description="Input tokens")
    output_tokens: int = Field(..., description="Output tokens")
    operation_type: str = Field(..., description="Operation type")
    latency_ms: Optional[int] = Field(None, description="Latency in ms")


class UsageReport(BaseModel):
    """Schema for usage reports"""
    user_id: str = Field(..., description="User's ID")
    period_start: datetime = Field(..., description="Report period start")
    period_end: datetime = Field(..., description="Report period end")
    
    # Summary statistics
    total_documents: int = Field(..., description="Total documents processed")
    total_pages: int = Field(..., description="Total pages processed")
    total_llm_calls: int = Field(..., description="Total LLM calls")
    total_tokens: int = Field(..., description="Total tokens used")
    estimated_total_cost_usd: Decimal = Field(..., description="Estimated total cost")
    
    # Breakdowns
    document_types: Dict[str, int] = Field(..., description="Documents by type")
    model_usage: Dict[str, Dict[str, Any]] = Field(..., description="Usage by model")
    daily_usage: List[Dict[str, Any]] = Field(..., description="Daily usage breakdown")
    
    # Performance metrics
    avg_processing_time_ms: Optional[int] = Field(None, description="Average processing time")
    success_rate: Optional[Decimal] = Field(None, description="Overall success rate")


class LimitCheckResult(BaseModel):
    """Schema for limit check results"""
    within_limits: bool = Field(..., description="Whether user is within limits")
    tier_name: str = Field(..., description="User's current tier")
    
    # Current usage
    current_documents: int = Field(..., description="Current document count")
    current_pages: int = Field(..., description="Current page count")
    current_tokens: int = Field(..., description="Current token count")
    current_cost_usd: Decimal = Field(..., description="Current cost")
    
    # Limits
    document_limit: Optional[int] = Field(None, description="Document limit")
    page_limit: Optional[int] = Field(None, description="Page limit")
    token_limit: Optional[int] = Field(None, description="Token limit")
    cost_limit_usd: Optional[Decimal] = Field(None, description="Cost limit")
    
    # Remaining
    remaining_documents: Optional[int] = Field(None, description="Remaining document capacity")
    remaining_pages: Optional[int] = Field(None, description="Remaining page capacity")
    remaining_tokens: Optional[int] = Field(None, description="Remaining token capacity")
    remaining_cost_usd: Optional[Decimal] = Field(None, description="Remaining cost capacity")
    
    # Warnings
    warnings: List[str] = Field(default=[], description="Usage warnings")
    upgrade_recommended: bool = Field(default=False, description="Whether upgrade is recommended")