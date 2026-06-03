-- Multi-provider seed: openai / anthropic / ollama providers + their current
-- models. Also refreshes the Gemini list with the 2.5 family.
--
-- Idempotent — ai_providers has UNIQUE(name) and ai_models has
-- UNIQUE(provider_id, name), so the ON CONFLICT clauses make re-runs safe.
--
-- Unblocks graphora-fe#15 (UI provider selector). Companion to the
-- AIConfigService refactor + generic /ai-config/{provider} endpoints in
-- this same change.

-- ─── Providers ─────────────────────────────────────────────────────────

INSERT INTO ai_providers (name, display_name, is_active) VALUES
    ('openai',    'OpenAI',           TRUE),
    ('anthropic', 'Anthropic Claude', TRUE),
    ('ollama',    'Ollama (self-hosted)', TRUE)
ON CONFLICT (name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    is_active = TRUE,
    updated_at = NOW();

-- ─── OpenAI models ─────────────────────────────────────────────────────

INSERT INTO ai_models (provider_id, name, display_name, version, is_active)
SELECT p.id, v.model_name, v.display_name, v.version, TRUE
FROM ai_providers p
JOIN (
    VALUES
        ('gpt-4o',         'GPT-4o',         'latest'),
        ('gpt-4o-mini',    'GPT-4o mini',    'latest'),
        ('gpt-4-turbo',    'GPT-4 Turbo',    'latest'),
        ('o1',             'o1 (reasoning)', 'latest'),
        ('o1-mini',        'o1 mini',        'latest'),
        ('o3-mini',        'o3 mini',        'latest')
) AS v(model_name, display_name, version) ON TRUE
WHERE p.name = 'openai'
ON CONFLICT (provider_id, name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    version      = EXCLUDED.version,
    is_active    = TRUE,
    updated_at   = NOW();

-- ─── Anthropic models ─────────────────────────────────────────────────

INSERT INTO ai_models (provider_id, name, display_name, version, is_active)
SELECT p.id, v.model_name, v.display_name, v.version, TRUE
FROM ai_providers p
JOIN (
    VALUES
        ('claude-opus-4-8',            'Claude Opus 4.8',   'latest'),
        ('claude-sonnet-4-6',          'Claude Sonnet 4.6', 'latest'),
        ('claude-haiku-4-5-20251001',  'Claude Haiku 4.5',  'latest'),
        ('claude-opus-4-7',            'Claude Opus 4.7',   'previous'),
        ('claude-sonnet-4-5',          'Claude Sonnet 4.5', 'previous')
) AS v(model_name, display_name, version) ON TRUE
WHERE p.name = 'anthropic'
ON CONFLICT (provider_id, name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    version      = EXCLUDED.version,
    is_active    = TRUE,
    updated_at   = NOW();

-- ─── Ollama models (self-hosted) ──────────────────────────────────────

INSERT INTO ai_models (provider_id, name, display_name, version, is_active)
SELECT p.id, v.model_name, v.display_name, v.version, TRUE
FROM ai_providers p
JOIN (
    VALUES
        ('llama3.3:70b',     'Llama 3.3 70B',     'latest'),
        ('llama3.2:3b',      'Llama 3.2 3B',      'latest'),
        ('qwen2.5:72b',      'Qwen 2.5 72B',      'latest'),
        ('qwen2.5:32b',      'Qwen 2.5 32B',      'latest'),
        ('deepseek-r1:70b',  'DeepSeek R1 70B',   'latest'),
        ('deepseek-r1:32b',  'DeepSeek R1 32B',   'latest'),
        ('mistral:7b',       'Mistral 7B',        'latest'),
        ('mixtral:8x7b',     'Mixtral 8x7B',      'latest')
) AS v(model_name, display_name, version) ON TRUE
WHERE p.name = 'ollama'
ON CONFLICT (provider_id, name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    version      = EXCLUDED.version,
    is_active    = TRUE,
    updated_at   = NOW();

-- ─── Gemini refresh — add 2.5 family on top of migration 6's 2.0 set ──

INSERT INTO ai_models (provider_id, name, display_name, version, is_active)
SELECT p.id, v.model_name, v.display_name, v.version, TRUE
FROM ai_providers p
JOIN (
    VALUES
        ('gemini-2.5-pro',   'Gemini 2.5 Pro',   'latest'),
        ('gemini-2.5-flash', 'Gemini 2.5 Flash', 'latest')
) AS v(model_name, display_name, version) ON TRUE
WHERE p.name = 'gemini'
ON CONFLICT (provider_id, name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    version      = EXCLUDED.version,
    is_active    = TRUE,
    updated_at   = NOW();
