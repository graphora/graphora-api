ALTER TABLE document_usage
    ADD COLUMN IF NOT EXISTS error_message TEXT;
