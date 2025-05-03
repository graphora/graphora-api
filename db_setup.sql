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