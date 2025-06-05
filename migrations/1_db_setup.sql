CREATE TABLE resolutions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ontology_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    previous_props JSONB NOT NULL,
    changed_props JSONB NOT NULL,
    resolved_props JSONB NOT NULL,
    resolution TEXT NOT NULL,
    learning_comment TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE change_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    merge_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    prod_node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    previous_props JSONB NOT NULL,
    changed_props JSONB NOT NULL,
    need_human_review BOOLEAN DEFAULT FALSE
);

CREATE TABLE merge_status (
    merge_id UUID PRIMARY KEY NOT NULL,
    transform_id TEXT NOT NULL,
    ontology_id TEXT NOT NULL,
    status TEXT NOT NULL,
    statistics JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    error TEXT
);

-- Audit Trail table for tracking all major operations
CREATE TABLE audit_trail (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    operation_type TEXT NOT NULL, -- 'ontology_stored', 'transform_started', 'transform_completed', 'merge_started', 'merge_completed'
    operation_id TEXT NOT NULL, -- ontology_id, transform_id, or merge_id
    resource_name TEXT, -- name/title of the resource
    status TEXT NOT NULL, -- 'success', 'failed', 'in_progress'
    metadata JSONB, -- additional operation-specific data
    error_message TEXT, -- if status is 'failed'
    duration_ms INTEGER, -- operation duration in milliseconds
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient querying
CREATE INDEX idx_audit_trail_user_id ON audit_trail(user_id);
CREATE INDEX idx_audit_trail_operation_type ON audit_trail(operation_type);
CREATE INDEX idx_audit_trail_created_at ON audit_trail(created_at);
CREATE INDEX idx_audit_trail_user_operation ON audit_trail(user_id, operation_type);

-- Ontology Storage table for Supabase instead of file storage
CREATE TABLE ontologies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    name TEXT,
    file_name TEXT,
    description TEXT,
    yaml_content TEXT NOT NULL,
    version TEXT DEFAULT '1.0.0',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient querying
CREATE INDEX idx_ontologies_user_id ON ontologies(user_id);
CREATE INDEX idx_ontologies_active ON ontologies(is_active) WHERE is_active = true;