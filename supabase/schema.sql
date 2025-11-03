-- Supabase schema for Graphora API backend
-- Run this in your Supabase project before starting the API.

create extension if not exists "uuid-ossp";
create extension if not exists "pgcrypto";

create table if not exists database_configs (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    uri text not null,
    username text not null,
    password text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists configs (
    id uuid primary key default gen_random_uuid(),
    user_id text not null unique,
    staging_db_id uuid not null references database_configs(id) on delete cascade,
    prod_db_id uuid not null references database_configs(id) on delete cascade,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists ai_providers (
    id uuid primary key default gen_random_uuid(),
    name text not null unique,
    display_name text not null,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists ai_models (
    id uuid primary key default gen_random_uuid(),
    provider_id uuid not null references ai_providers(id) on delete cascade,
    name text not null,
    display_name text not null,
    version text,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (provider_id, name)
);

create table if not exists ai_provider_configs (
    id uuid primary key default gen_random_uuid(),
    provider_id uuid not null references ai_providers(id) on delete cascade,
    model_id uuid not null references ai_models(id) on delete cascade,
    api_key text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists user_ai_configs (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    provider_config_id uuid not null references ai_provider_configs(id) on delete cascade,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (user_id)
);

create table if not exists audit_trail (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    operation_type text not null,
    operation_id text,
    resource_name text,
    status text not null,
    metadata jsonb default '{}'::jsonb,
    duration_ms integer,
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists document_usage (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    transform_id text not null,
    document_name text not null,
    document_type text not null,
    document_size_bytes bigint not null,
    page_count integer not null default 0,
    processing_status text not null,
    processing_started_at timestamptz not null,
    processing_completed_at timestamptz,
    processing_duration_ms integer,
    success_rate numeric,
    is_reprocessing boolean not null default false,
    reprocessing_reason text,
    chunks_created integer not null default 0,
    nodes_extracted integer not null default 0,
    relationships_extracted integer not null default 0,
    billable_pages integer not null default 0,
    billable_processing_units integer not null default 1,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists llm_usage (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    transform_id text,
    document_usage_id uuid references document_usage(id) on delete set null,
    model_provider text not null,
    model_name text not null,
    model_version text,
    input_tokens integer not null default 0,
    output_tokens integer not null default 0,
    total_tokens integer not null default 0,
    estimated_cost_usd numeric,
    cost_per_1k_input_tokens numeric,
    cost_per_1k_output_tokens numeric,
    operation_type text not null,
    operation_context text,
    latency_ms integer,
    success boolean not null default true,
    error_message text,
    request_timestamp timestamptz not null,
    response_timestamp timestamptz,
    created_at timestamptz not null default now()
);

create table if not exists usage_aggregates (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    period_type text not null,
    period_start timestamptz not null,
    period_end timestamptz not null,
    total_documents integer not null default 0,
    total_pages integer not null default 0,
    total_processing_time_ms bigint not null default 0,
    avg_pages_per_document numeric,
    success_rate numeric,
    reprocessing_count integer not null default 0,
    pdf_documents integer not null default 0,
    docx_documents integer not null default 0,
    txt_documents integer not null default 0,
    other_documents integer not null default 0,
    total_llm_calls integer not null default 0,
    total_input_tokens bigint not null default 0,
    total_output_tokens bigint not null default 0,
    total_estimated_cost_usd numeric,
    model_usage_breakdown jsonb,
    avg_tokens_per_page numeric,
    avg_processing_time_per_page_ms integer,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists entity_ledger (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    entity_type text not null,
    canonical_key text not null,
    canonical_id text not null,
    features jsonb default '{}'::jsonb,
    confidence numeric,
    first_seen_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (user_id, entity_type, canonical_key)
);

create table if not exists pricing_tiers (
    id uuid primary key default gen_random_uuid(),
    tier_name text not null unique,
    description text,
    monthly_document_limit integer,
    monthly_page_limit integer,
    monthly_token_limit integer,
    monthly_cost_limit_usd numeric,
    features jsonb,
    base_price_usd numeric not null default 0,
    price_per_page_usd numeric,
    price_per_1k_tokens_usd numeric,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists user_pricing_tiers (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    tier_id uuid not null references pricing_tiers(id) on delete cascade,
    billing_period_start timestamptz not null,
    billing_period_end timestamptz not null,
    current_documents integer not null default 0,
    current_pages integer not null default 0,
    current_tokens bigint not null default 0,
    current_cost_usd numeric not null default 0,
    is_active boolean not null default true,
    over_limit boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_document_usage_user on document_usage(user_id);
create index if not exists idx_llm_usage_user on llm_usage(user_id);
create index if not exists idx_audit_trail_user on audit_trail(user_id);
create index if not exists idx_user_ai_configs_user on user_ai_configs(user_id);
