-- Model pricing configuration tables
-- Run this in your Supabase SQL editor

-- Model providers table
CREATE TABLE IF NOT EXISTS model_providers (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    provider_name VARCHAR(50) NOT NULL UNIQUE,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model pricing table
CREATE TABLE IF NOT EXISTS model_pricing (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    provider_id UUID NOT NULL REFERENCES model_providers(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    
    -- Pricing per 1K tokens in USD
    input_price_per_1k_tokens DECIMAL(10,6) NOT NULL,
    output_price_per_1k_tokens DECIMAL(10,6) NOT NULL,
    
    -- Additional metadata
    model_context_window INTEGER,
    model_description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique model names per provider
    CONSTRAINT unique_provider_model UNIQUE (provider_id, model_name)
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_model_providers_name ON model_providers(provider_name);
CREATE INDEX IF NOT EXISTS idx_model_providers_active ON model_providers(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_model_pricing_provider ON model_pricing(provider_id);
CREATE INDEX IF NOT EXISTS idx_model_pricing_model_name ON model_pricing(model_name);
CREATE INDEX IF NOT EXISTS idx_model_pricing_active ON model_pricing(is_active) WHERE is_active = TRUE;

-- Create triggers for updated_at
CREATE TRIGGER update_model_providers_updated_at 
    BEFORE UPDATE ON model_providers 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_pricing_updated_at 
    BEFORE UPDATE ON model_pricing 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default model providers
INSERT INTO model_providers (provider_name, display_name, description) VALUES
    ('gemini', 'Google Gemini', 'Google''s large language models'),
    ('openai', 'OpenAI', 'OpenAI GPT models'),
    ('anthropic', 'Anthropic', 'Anthropic Claude models'),
    ('baml', 'BAML', 'BAML proxy for various models')
ON CONFLICT (provider_name) DO NOTHING;

-- Insert default model pricing
-- First, insert Gemini models
INSERT INTO model_pricing (provider_id, model_name, input_price_per_1k_tokens, output_price_per_1k_tokens, model_description)
SELECT 
    p.id,
    v.model_name,
    v.input_price,
    v.output_price,
    v.description
FROM model_providers p
CROSS JOIN (VALUES
    ('gemini-2.0-flash-lite-001', 0.075, 0.30, 'Fast and efficient Gemini model'),
    ('gemini-1.5-pro', 3.50, 10.50, 'Most capable Gemini model'),
    ('gemini-1.5-flash', 0.075, 0.30, 'Fast Gemini model for high-volume tasks')
) AS v(model_name, input_price, output_price, description)
WHERE p.provider_name = 'gemini'
ON CONFLICT (provider_id, model_name) DO NOTHING;

-- Insert OpenAI models
INSERT INTO model_pricing (provider_id, model_name, input_price_per_1k_tokens, output_price_per_1k_tokens, model_description)
SELECT 
    p.id,
    v.model_name,
    v.input_price,
    v.output_price,
    v.description
FROM model_providers p
CROSS JOIN (VALUES
    ('gpt-4', 30.00, 60.00, 'Most capable GPT-4 model'),
    ('gpt-4-turbo', 10.00, 30.00, 'Faster GPT-4 variant'),
    ('gpt-3.5-turbo', 0.50, 1.50, 'Fast and cost-effective model')
) AS v(model_name, input_price, output_price, description)
WHERE p.provider_name = 'openai'
ON CONFLICT (provider_id, model_name) DO NOTHING;

-- Insert Anthropic models
INSERT INTO model_pricing (provider_id, model_name, input_price_per_1k_tokens, output_price_per_1k_tokens, model_description)
SELECT 
    p.id,
    v.model_name,
    v.input_price,
    v.output_price,
    v.description
FROM model_providers p
CROSS JOIN (VALUES
    ('claude-3-opus', 15.00, 75.00, 'Most powerful Claude model'),
    ('claude-3-sonnet', 3.00, 15.00, 'Balanced Claude model'),
    ('claude-3-haiku', 0.25, 1.25, 'Fast and cost-effective Claude model')
) AS v(model_name, input_price, output_price, description)
WHERE p.provider_name = 'anthropic'
ON CONFLICT (provider_id, model_name) DO NOTHING;

-- Insert BAML models (proxy models)
INSERT INTO model_pricing (provider_id, model_name, input_price_per_1k_tokens, output_price_per_1k_tokens, model_description)
SELECT 
    p.id,
    v.model_name,
    v.input_price,
    v.output_price,
    v.description
FROM model_providers p
CROSS JOIN (VALUES
    ('unknown', 1.00, 3.00, 'Default BAML pricing for unknown models'),
    ('gpt-4', 30.00, 60.00, 'GPT-4 via BAML proxy'),
    ('gpt-3.5-turbo', 0.50, 1.50, 'GPT-3.5 Turbo via BAML proxy'),
    ('claude-3-sonnet', 3.00, 15.00, 'Claude-3 Sonnet via BAML proxy'),
    ('claude-3-haiku', 0.25, 1.25, 'Claude-3 Haiku via BAML proxy'),
    ('gemini-1.5-flash', 0.075, 0.30, 'Gemini 1.5 Flash via BAML proxy'),
    ('gemini-2.0-flash-lite-001', 0.075, 0.30, 'Gemini 2.0 Flash Lite via BAML proxy')
) AS v(model_name, input_price, output_price, description)
WHERE p.provider_name = 'baml'
ON CONFLICT (provider_id, model_name) DO NOTHING; 