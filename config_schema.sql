-- Configuration tables for Graphora user database settings
-- Run this in your Supabase SQL editor

-- Create database_configs table
CREATE TABLE IF NOT EXISTS database_configs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    uri VARCHAR(500) NOT NULL,
    username VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create configs table
CREATE TABLE IF NOT EXISTS configs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL UNIQUE,
    staging_db_id UUID NOT NULL REFERENCES database_configs(id) ON DELETE CASCADE,
    prod_db_id UUID NOT NULL REFERENCES database_configs(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure staging and prod databases are different
    CONSTRAINT different_databases CHECK (staging_db_id != prod_db_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_configs_user_email ON configs(user_email);
CREATE INDEX IF NOT EXISTS idx_database_configs_uri ON database_configs(uri);
CREATE INDEX IF NOT EXISTS idx_configs_staging_db ON configs(staging_db_id);
CREATE INDEX IF NOT EXISTS idx_configs_prod_db ON configs(prod_db_id);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at
CREATE TRIGGER update_database_configs_updated_at 
    BEFORE UPDATE ON database_configs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configs_updated_at 
    BEFORE UPDATE ON configs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add Row Level Security (RLS) policies if needed
-- ALTER TABLE configs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE database_configs ENABLE ROW LEVEL SECURITY;

-- Example RLS policy (uncomment if you want to restrict access by user)
-- CREATE POLICY "Users can only access their own configs" ON configs
--     FOR ALL USING (user_email = auth.email());

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT ALL ON configs TO authenticated;
-- GRANT ALL ON database_configs TO authenticated;
-- GRANT USAGE ON SEQUENCE configs_id_seq TO authenticated;
-- GRANT USAGE ON SEQUENCE database_configs_id_seq TO authenticated; 