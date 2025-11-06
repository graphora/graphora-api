-- Usage tracking tables for API pricing and analytics
-- Run this in your Supabase SQL editor

-- Document processing usage tracking
CREATE TABLE IF NOT EXISTS document_usage (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    transform_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    
    -- Document details
    document_name VARCHAR(500) NOT NULL,
    document_type VARCHAR(50) NOT NULL, -- PDF, DOCX, TXT, etc.
    document_size_bytes BIGINT NOT NULL,
    page_count INTEGER DEFAULT 0,
    
    -- Processing details
    processing_status VARCHAR(50) NOT NULL, -- success, failed, partial
    processing_started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_duration_ms INTEGER,
    
    -- Quality metrics
    success_rate DECIMAL(5,2), -- percentage
    is_reprocessing BOOLEAN DEFAULT FALSE,
    reprocessing_reason VARCHAR(500),
    
    -- Storage and chunking
    chunks_created INTEGER DEFAULT 0,
    nodes_extracted INTEGER DEFAULT 0,
    relationships_extracted INTEGER DEFAULT 0,
    
    -- Pricing relevant
    billable_pages INTEGER DEFAULT 0,
    billable_processing_units INTEGER DEFAULT 1,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- LLM usage tracking for costs
CREATE TABLE IF NOT EXISTS llm_usage (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    transform_id VARCHAR(255),
    document_usage_id UUID REFERENCES document_usage(id) ON DELETE CASCADE,
    
    -- LLM call details
    model_provider VARCHAR(50) NOT NULL, -- gemini, openai, anthropic, etc.
    model_name VARCHAR(100) NOT NULL, -- gpt-4, gemini-pro, etc.
    model_version VARCHAR(50),
    
    -- Token usage
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    
    -- Cost tracking
    estimated_cost_usd DECIMAL(10,6), -- Cost in USD
    cost_per_1k_input_tokens DECIMAL(8,6),
    cost_per_1k_output_tokens DECIMAL(8,6),
    
    -- Operation details
    operation_type VARCHAR(100) NOT NULL, -- entity_extraction, relationship_extraction, etc.
    operation_context VARCHAR(500), -- chunk_processing, pdf_analysis, etc.
    
    -- Performance metrics
    latency_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    
    -- Request details
    request_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    response_timestamp TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Aggregated usage statistics per user/organization
CREATE TABLE IF NOT EXISTS usage_aggregates (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    
    -- Time period
    period_type VARCHAR(20) NOT NULL, -- daily, weekly, monthly
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Document processing aggregates
    total_documents INTEGER DEFAULT 0,
    total_pages INTEGER DEFAULT 0,
    total_processing_time_ms BIGINT DEFAULT 0,
    avg_pages_per_document DECIMAL(8,2),
    success_rate DECIMAL(5,2),
    reprocessing_count INTEGER DEFAULT 0,
    
    -- Document type breakdown
    pdf_documents INTEGER DEFAULT 0,
    docx_documents INTEGER DEFAULT 0,
    txt_documents INTEGER DEFAULT 0,
    other_documents INTEGER DEFAULT 0,
    
    -- LLM usage aggregates
    total_llm_calls INTEGER DEFAULT 0,
    total_input_tokens BIGINT DEFAULT 0,
    total_output_tokens BIGINT DEFAULT 0,
    total_estimated_cost_usd DECIMAL(12,6),
    
    -- Model usage breakdown (JSON for flexibility)
    model_usage_breakdown JSONB,
    
    -- Performance metrics
    avg_tokens_per_page DECIMAL(8,2),
    avg_processing_time_per_page_ms INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique periods per user
    CONSTRAINT unique_user_period UNIQUE (user_id, period_type, period_start)
);

-- Pricing tiers and limits
CREATE TABLE IF NOT EXISTS pricing_tiers (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    tier_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    
    -- Document limits
    monthly_document_limit INTEGER,
    monthly_page_limit INTEGER,
    
    -- LLM limits
    monthly_token_limit BIGINT,
    monthly_cost_limit_usd DECIMAL(10,2),
    
    -- Features
    features JSONB, -- JSON array of feature names
    
    -- Pricing
    base_price_usd DECIMAL(10,2) DEFAULT 0,
    price_per_page_usd DECIMAL(8,6),
    price_per_1k_tokens_usd DECIMAL(8,6),
    
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User tier assignments
CREATE TABLE IF NOT EXISTS user_pricing_tiers (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL UNIQUE,
    tier_id UUID NOT NULL REFERENCES pricing_tiers(id),
    
    -- Billing period
    billing_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    billing_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Current usage against limits
    current_documents INTEGER DEFAULT 0,
    current_pages INTEGER DEFAULT 0,
    current_tokens BIGINT DEFAULT 0,
    current_cost_usd DECIMAL(10,6) DEFAULT 0,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    over_limit BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_document_usage_user_id ON document_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_document_usage_transform_id ON document_usage(transform_id);
CREATE INDEX IF NOT EXISTS idx_document_usage_session_id ON document_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_document_usage_created_at ON document_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_document_usage_status ON document_usage(processing_status);

CREATE INDEX IF NOT EXISTS idx_llm_usage_user_id ON llm_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_transform_id ON llm_usage(transform_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_model ON llm_usage(model_provider, model_name);
CREATE INDEX IF NOT EXISTS idx_llm_usage_timestamp ON llm_usage(request_timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_usage_document_id ON llm_usage(document_usage_id);

CREATE INDEX IF NOT EXISTS idx_usage_aggregates_user_period ON usage_aggregates(user_id, period_type, period_start);

CREATE INDEX IF NOT EXISTS idx_user_pricing_tiers_user_id ON user_pricing_tiers(user_id);
CREATE INDEX IF NOT EXISTS idx_user_pricing_tiers_active ON user_pricing_tiers(is_active) WHERE is_active = true;

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_document_usage_updated_at 
    BEFORE UPDATE ON document_usage 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_usage_aggregates_updated_at 
    BEFORE UPDATE ON usage_aggregates 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pricing_tiers_updated_at 
    BEFORE UPDATE ON pricing_tiers 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_pricing_tiers_updated_at 
    BEFORE UPDATE ON user_pricing_tiers 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default pricing tiers
INSERT INTO pricing_tiers (tier_name, description, monthly_document_limit, monthly_page_limit, monthly_token_limit, 
                          base_price_usd, price_per_page_usd, price_per_1k_tokens_usd, features)
VALUES 
    ('Starter', 'Small business tier', 100, 2000, 5000000, 59.99, 0.01, 0.002, 
     '["advanced_extraction", "pdf_support", "docx_support", "email_support"]'::jsonb),
    ('Professional', 'Growing business tier', 500, 10000, 2000000, 99.99, 0.008, 0.0015, 
     '["advanced_extraction", "all_formats", "priority_support"]'::jsonb),
    ('Enterprise', 'Large scale operations', 10000, null, null, 1999.99, 0.005, 0.001, 
     '["enterprise_extraction", "unlimited_processing", "dedicated_support", "custom_models"]'::jsonb)
ON CONFLICT (tier_name) DO NOTHING;
