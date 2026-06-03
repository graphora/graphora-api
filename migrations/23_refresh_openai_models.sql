-- OpenAI catalog refresh — addresses PR #24 review Medium.
--
-- Migration 22 seeded an OpenAI catalog that is now stale per OpenAI's
-- official model list (https://developers.openai.com/api/docs/models/all
-- as of 2026-06-03):
--
--   * gpt-4-turbo, o1, o1-mini, o3-mini — all deprecated; OpenAI's
--     current "All models" page no longer lists them.
--   * gpt-5.x family (the GPT-5 generation introduced in 2025) is
--     absent from the seed entirely.
--
-- This migration:
--   1. Deactivates the four deprecated rows (is_active = FALSE), so
--      they stop appearing in the UI dropdown but stay in the DB for
--      historical reference / any in-flight user configs that still
--      point at them (those configs continue to work — only the
--      catalog visibility changes).
--   2. Inserts the current GPT-5.x family + o3 / o3-pro + the still-
--      active gpt-4.1 family.
--
-- Idempotent via ON CONFLICT (provider_id, name) DO UPDATE.

-- ─── Deactivate deprecated rows ────────────────────────────────────────

UPDATE ai_models m
SET is_active = FALSE,
    updated_at = NOW()
FROM ai_providers p
WHERE m.provider_id = p.id
  AND p.name = 'openai'
  AND m.name IN ('gpt-4-turbo', 'o1', 'o1-mini', 'o3-mini');

-- ─── Insert current models ────────────────────────────────────────────

INSERT INTO ai_models (provider_id, name, display_name, version, is_active)
SELECT p.id, v.model_name, v.display_name, v.version, TRUE
FROM ai_providers p
JOIN (
    VALUES
        -- GPT-5.5 family (flagship — Spring 2026)
        ('gpt-5.5',         'GPT-5.5',          'latest'),
        ('gpt-5.5-pro',     'GPT-5.5 Pro',      'latest'),
        -- GPT-5.4 family (cost-tier workhorses)
        ('gpt-5.4',         'GPT-5.4',          'latest'),
        ('gpt-5.4-pro',     'GPT-5.4 Pro',      'latest'),
        ('gpt-5.4-mini',    'GPT-5.4 mini',     'latest'),
        ('gpt-5.4-nano',    'GPT-5.4 nano',     'latest'),
        -- GPT-5 base family (previous generation, still available)
        ('gpt-5',           'GPT-5',            'previous'),
        ('gpt-5-mini',      'GPT-5 mini',       'previous'),
        ('gpt-5-nano',      'GPT-5 nano',       'previous'),
        -- GPT-4.1 family (non-reasoning, still listed as active)
        ('gpt-4.1',         'GPT-4.1',          'previous'),
        ('gpt-4.1-mini',    'GPT-4.1 mini',     'previous'),
        -- o-series reasoning (o1/o1-mini/o3-mini deprecated above)
        ('o3',              'o3 (reasoning)',   'latest'),
        ('o3-pro',          'o3 Pro (reasoning)', 'latest')
) AS v(model_name, display_name, version) ON TRUE
WHERE p.name = 'openai'
ON CONFLICT (provider_id, name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    version      = EXCLUDED.version,
    is_active    = TRUE,
    updated_at   = NOW();
