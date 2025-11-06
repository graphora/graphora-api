alter table if exists document_usage
    add column if not exists session_id text;

create index if not exists idx_document_usage_session
    on document_usage(session_id);
