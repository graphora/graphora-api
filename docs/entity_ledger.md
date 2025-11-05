# Entity Ledger

Entity fingerprints (canonical key + ID + features) are stored in the `entity_ledger` table so repeat transforms can reuse past matches. Each entry is scoped by `user_id` and `entity_type`.

Schema:
```sql
create table entity_ledger (
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
```

### Usage Flow
1. During extraction, nodes receive a `canonical_key` (built from ontology-driven canonical properties) and a deterministic `canonical_id` (_uuid5 over the key_).
2. `EntityLedgerService.hydrate_nodes()` looks up existing entries for `(user_id, entity_type, canonical_key)` and reuses the stored `canonical_id` before entity resolution.
3. After dedupe, `record_nodes()` upserts the fingerprints for future transforms.

You can plug in Supabase credentials or rely on the in-memory store (useful for tests). Custom canonicalisation logic comes from the ontology metadata or by registering a canonicalizer at runtime.
