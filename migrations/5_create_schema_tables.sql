-- Migration: Create schema generation tables
-- Version: 001
-- Description: Create tables for storing generated schemas, embeddings, and usage analytics
CREATE EXTENSION IF NOT EXISTS vector;
-- Create generated_schemas table
CREATE TABLE IF NOT EXISTS generated_schemas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    content TEXT NOT NULL,
    domain VARCHAR(100) NOT NULL,
    tags TEXT[] DEFAULT '{}',
    confidence DECIMAL(3,2) DEFAULT 0.8,
    context JSONB DEFAULT '{}',
    is_public BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_generated_schemas_user_id ON generated_schemas(user_id);
CREATE INDEX IF NOT EXISTS idx_generated_schemas_domain ON generated_schemas(domain);
CREATE INDEX IF NOT EXISTS idx_generated_schemas_is_public ON generated_schemas(is_public);
CREATE INDEX IF NOT EXISTS idx_generated_schemas_tags ON generated_schemas USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_generated_schemas_usage_count ON generated_schemas(usage_count DESC);
CREATE INDEX IF NOT EXISTS idx_generated_schemas_created_at ON generated_schemas(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_generated_schemas_text_search ON generated_schemas USING GIN(to_tsvector('english', title || ' ' || description));

-- Create schema_usage_events table for analytics
CREATE TABLE IF NOT EXISTS schema_usage_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_id UUID NOT NULL REFERENCES generated_schemas(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for usage events
CREATE INDEX IF NOT EXISTS idx_schema_usage_events_schema_id ON schema_usage_events(schema_id);
CREATE INDEX IF NOT EXISTS idx_schema_usage_events_user_id ON schema_usage_events(user_id);
CREATE INDEX IF NOT EXISTS idx_schema_usage_events_event_type ON schema_usage_events(event_type);
CREATE INDEX IF NOT EXISTS idx_schema_usage_events_timestamp ON schema_usage_events(timestamp DESC);

-- Create schema_embeddings table for vector search (future use)
CREATE TABLE IF NOT EXISTS schema_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_id UUID NOT NULL REFERENCES generated_schemas(id) ON DELETE CASCADE,
    embedding_type VARCHAR(50) NOT NULL DEFAULT 'content',
    embedding VECTOR(1536), -- OpenAI embedding size, adjust as needed
    model_version VARCHAR(50) NOT NULL DEFAULT 'gemini-embedding-exp-03-07',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(schema_id, embedding_type)
);

-- Create indexes for embeddings (if pgvector extension is available)
-- Note: This requires the pgvector extension to be installed
-- CREATE INDEX IF NOT EXISTS idx_schema_embeddings_vector ON schema_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for updated_at
DROP TRIGGER IF EXISTS update_generated_schemas_updated_at ON generated_schemas;
CREATE TRIGGER update_generated_schemas_updated_at
    BEFORE UPDATE ON generated_schemas
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to increment schema usage count atomically
CREATE OR REPLACE FUNCTION increment_schema_usage(schema_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE generated_schemas 
    SET usage_count = usage_count + 1 
    WHERE id = schema_id;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Insert some sample public schemas for testing
INSERT INTO generated_schemas (id, user_id, title, description, content, domain, tags, is_public, usage_count) VALUES
(
    gen_random_uuid(),
    'system',
    'Basic Business Schema',
    'A simple business schema for organizations, people, and relationships',
    'version: 0.1.0
entities:
  Organization:
    properties:
      name:
        type: str
        description: Organization name
        unique: true
        required: true
      type:
        type: str
        description: Organization type
        required: true
    relationships:
      HAS_EMPLOYEE:
        target: Person
  Person:
    properties:
      name:
        type: str
        description: Person full name
        required: true
      email:
        type: str
        description: Email address
        unique: true',
    'Business/Enterprise',
    ARRAY['business', 'organization', 'person', 'employee'],
    true,
    25
),
(
    gen_random_uuid(),
    'system',
    'Healthcare Patient Schema',
    'Schema for managing patient information and medical records',
    'version: 0.1.0
entities:
  Patient:
    properties:
      patient_id:
        type: str
        description: Unique patient identifier
        unique: true
        required: true
      name:
        type: str
        description: Patient name
        required: true
      date_of_birth:
        type: date
        description: Patient date of birth
    relationships:
      HAS_RECORD:
        target: MedicalRecord
  MedicalRecord:
    properties:
      record_id:
        type: str
        description: Record identifier
        unique: true
        required: true
      diagnosis:
        type: str
        description: Medical diagnosis',
    'Healthcare/Medical',
    ARRAY['healthcare', 'patient', 'medical', 'records'],
    true,
    18
),
(
    gen_random_uuid(),
    'system',
    'E-commerce Product Catalog',
    'Schema for e-commerce product management and categories',
    'version: 0.1.0
entities:
  Product:
    properties:
      sku:
        type: str
        description: Product SKU
        unique: true
        required: true
      name:
        type: str
        description: Product name
        required: true
      price:
        type: float
        description: Product price
    relationships:
      BELONGS_TO:
        target: Category
  Category:
    properties:
      name:
        type: str
        description: Category name
        unique: true
        required: true',
    'E-commerce/Retail',
    ARRAY['ecommerce', 'product', 'catalog', 'retail'],
    true,
    32
);

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON generated_schemas TO your_app_user;
-- GRANT ALL PRIVILEGES ON schema_usage_events TO your_app_user;
-- GRANT ALL PRIVILEGES ON schema_embeddings TO your_app_user;