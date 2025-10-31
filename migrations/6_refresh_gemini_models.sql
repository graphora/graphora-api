-- Upsert the current Google Gemini models and pricing (Q1 2025)

-- Insert or update Gemini models available to end users
INSERT INTO ai_models (provider_id, name, display_name, version, is_active)
SELECT
    p.id,
    v.model_name,
    v.display_name,
    v.version,
    TRUE
FROM ai_providers p
JOIN (
    VALUES
        ('gemini-2.0-pro', 'Gemini 2.0 Pro', 'latest'),
        ('gemini-2.0-flash', 'Gemini 2.0 Flash', 'latest'),
        ('gemini-2.0-flash-thinking', 'Gemini 2.0 Flash Thinking', 'latest'),
        ('gemini-2.0-flash-lite', 'Gemini 2.0 Flash Lite', 'latest'),
        ('gemini-1.5-pro-latest', 'Gemini 1.5 Pro (latest)', 'latest'),
        ('gemini-1.5-flash-latest', 'Gemini 1.5 Flash (latest)', 'latest'),
        ('gemini-1.5-flash-8b-latest', 'Gemini 1.5 Flash 8B (latest)', 'latest')
) AS v(model_name, display_name, version)
    ON TRUE
WHERE p.name = 'gemini'
ON CONFLICT (provider_id, name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    version = EXCLUDED.version,
    is_active = TRUE,
    updated_at = NOW();

-- Keep pricing metadata in sync with the newly supported Gemini models
INSERT INTO model_pricing (
    provider_id,
    model_name,
    model_version,
    input_price_per_1k_tokens,
    output_price_per_1k_tokens,
    model_context_window,
    model_description,
    is_active
)
SELECT
    mp.id,
    v.model_name,
    v.version,
    v.input_price,
    v.output_price,
    v.context_window,
    v.description,
    TRUE
FROM model_providers mp
JOIN (
    VALUES
        ('gemini-2.0-pro', 'latest', 3.50, 10.50, 2000000, 'Most capable Gemini 2.0 Pro model'),
        ('gemini-2.0-flash', 'latest', 0.10, 0.40, 1000000, 'Balanced Gemini 2.0 Flash model'),
        ('gemini-2.0-flash-thinking', 'latest', 0.15, 0.60, 1000000, 'Gemini 2.0 Flash model with reasoning traces'),
        ('gemini-2.0-flash-lite', 'latest', 0.06, 0.24, 1000000, 'Lightweight Gemini 2.0 Flash Lite model'),
        ('gemini-1.5-pro-latest', 'latest', 3.50, 10.50, 2000000, 'Gemini 1.5 Pro latest channel'),
        ('gemini-1.5-flash-latest', 'latest', 0.075, 0.30, 1000000, 'Gemini 1.5 Flash latest channel'),
        ('gemini-1.5-flash-8b-latest', 'latest', 0.05, 0.20, 1000000, 'Gemini 1.5 Flash 8B latest channel')
) AS v(model_name, version, input_price, output_price, context_window, description)
    ON TRUE
WHERE mp.provider_name = 'gemini'
ON CONFLICT (provider_id, model_name) DO UPDATE SET
    model_version = EXCLUDED.model_version,
    input_price_per_1k_tokens = EXCLUDED.input_price_per_1k_tokens,
    output_price_per_1k_tokens = EXCLUDED.output_price_per_1k_tokens,
    model_context_window = EXCLUDED.model_context_window,
    model_description = EXCLUDED.model_description,
    is_active = TRUE,
    updated_at = NOW();
