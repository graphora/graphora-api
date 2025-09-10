-- Gemini provider configuration tables for Graphora
-- Run this in your Supabase SQL editor after running the previous migrations

-- Create ai_providers table for future extensibility (supporting multiple AI providers)
CREATE TABLE IF NOT EXISTS ai_providers (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE, -- e.g., 'gemini', 'openai', 'claude'
    display_name VARCHAR(255) NOT NULL, -- e.g., 'Google Gemini AI Studio', 'OpenAI', 'Anthropic Claude'
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create ai_models table to store available models for each provider
CREATE TABLE IF NOT EXISTS ai_models (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    provider_id UUID NOT NULL REFERENCES ai_providers(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL, -- e.g., 'gemini-1.5-pro', 'gemini-1.5-flash'
    display_name VARCHAR(255) NOT NULL, -- e.g., 'Gemini 1.5 Pro', 'Gemini 1.5 Flash'
    version VARCHAR(50), -- e.g., '001', '002'
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(provider_id, name)
);

-- Create ai_provider_configs table to store encrypted API keys and settings
CREATE TABLE IF NOT EXISTS ai_provider_configs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    provider_id UUID NOT NULL REFERENCES ai_providers(id) ON DELETE CASCADE,
    api_key VARCHAR(500) NOT NULL, -- Encrypted API key
    default_model_id UUID REFERENCES ai_models(id) ON DELETE SET NULL,
    config_data JSONB DEFAULT '{}', -- Additional provider-specific settings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create user_ai_configs table to link users with their AI provider configurations
CREATE TABLE IF NOT EXISTS user_ai_configs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL UNIQUE, -- User ID from authentication
    active_provider_config_id UUID NOT NULL REFERENCES ai_provider_configs(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default AI providers
INSERT INTO ai_providers (name, display_name, is_active) VALUES
    ('gemini', 'Google Gemini AI Studio', true)
ON CONFLICT (name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    updated_at = NOW();

INSERT INTO ai_models (provider_id, name, display_name, version, is_active) 
SELECT 
    p.id,
    'gemini-2.0-flash-lite',
    'Gemini 2.0 Flash Lite',
    'latest',
    true
FROM ai_providers p WHERE p.name = 'gemini'
ON CONFLICT (provider_id, name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    version = EXCLUDED.version,
    updated_at = NOW();

INSERT INTO ai_models (provider_id, name, display_name, version, is_active) 
SELECT 
    p.id,
    'gemini-2.5-flash-lite',
    'Gemini 2.5 Flash Lite',
    'latest',
    true
FROM ai_providers p WHERE p.name = 'gemini'
ON CONFLICT (provider_id, name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    version = EXCLUDED.version,
    updated_at = NOW();

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ai_models_provider_id ON ai_models(provider_id);
CREATE INDEX IF NOT EXISTS idx_ai_provider_configs_provider_id ON ai_provider_configs(provider_id);
CREATE INDEX IF NOT EXISTS idx_user_ai_configs_user_id ON user_ai_configs(user_id);
CREATE INDEX IF NOT EXISTS idx_user_ai_configs_provider_config_id ON user_ai_configs(active_provider_config_id);

-- Update triggers for updated_at columns
CREATE TRIGGER update_ai_providers_updated_at 
    BEFORE UPDATE ON ai_providers 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_models_updated_at 
    BEFORE UPDATE ON ai_models 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_provider_configs_updated_at 
    BEFORE UPDATE ON ai_provider_configs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_ai_configs_updated_at 
    BEFORE UPDATE ON user_ai_configs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add Row Level Security (RLS) policies if needed
-- ALTER TABLE ai_providers ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ai_models ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ai_provider_configs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE user_ai_configs ENABLE ROW LEVEL SECURITY;

-- Example RLS policy for user_ai_configs (uncomment if you want to restrict access by user)
-- CREATE POLICY "Users can only access their own AI configs" ON user_ai_configs
--     FOR ALL USING (user_id = auth.uid()::text);

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT ALL ON ai_providers TO authenticated;
-- GRANT ALL ON ai_models TO authenticated;
-- GRANT ALL ON ai_provider_configs TO authenticated;
-- GRANT ALL ON user_ai_configs TO authenticated; 