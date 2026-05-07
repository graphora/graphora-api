import pytest

from graphora_server.config import settings
from graphora_server.services.entity_ledger_service import EntityLedgerService
from graphora_server.services.transform.models import BaseNode
from graphora_server.services.transform.helpers import _make_canonical_node_id


@pytest.fixture(autouse=True)
def disable_database(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "POSTGRES_HOST", None)


@pytest.mark.asyncio
async def test_entity_ledger_memory_store_roundtrip():
    service = EntityLedgerService(memory_store={})

    canonical_key = "Company:name=acme"
    canonical_id = _make_canonical_node_id(canonical_key)

    node = BaseNode(
        id="local-1",
        type="Company",
        properties={"name": "Acme"},
        canonical_properties={"name": "acme"},
        canonical_key=canonical_key,
        canonical_id=canonical_id,
        confidence_score=0.9,
    )

    await service.hydrate_nodes("user-1", [node])  # nothing stored yet
    assert node.canonical_id == canonical_id

    await service.record_nodes("user-1", [node])

    node2 = BaseNode(
        id="local-2",
        type="Company",
        properties={"name": "ACME"},
        canonical_properties={"name": "acme"},
        canonical_key=canonical_key,
    )

    await service.hydrate_nodes("user-1", [node2])
    assert node2.canonical_id == canonical_id


@pytest.mark.asyncio
async def test_entity_ledger_ignore_missing_user():
    service = EntityLedgerService(memory_store={})

    node = BaseNode(type="Company")
    await service.hydrate_nodes(None, [node])
    await service.record_nodes(None, [node])
    # no exceptions


# --------------------------------------------------------------
# Slice 1: embedding persistence + stored-embedding similarity.
# Pin the new contract so a refactor can't quietly skip the
# embedding write or revert to the per-call recomputation that
# was the dominant cost pre-slice-1.
# --------------------------------------------------------------


def _node_with_canonical(name: str, canonical_key: str) -> BaseNode:
    """Helper: build a node with all the fields record_nodes needs.
    The canonical_id mirrors what _make_canonical_node_id produces
    so the dedup key stays stable across hydrate / record cycles."""
    return BaseNode(
        id=f"local-{name.lower()}",
        type="Company",
        properties={"name": name},
        canonical_properties={"name": name.lower()},
        canonical_key=canonical_key,
        canonical_id=_make_canonical_node_id(canonical_key),
        confidence_score=0.9,
    )


class _FakeEmbedder:
    """Minimal stand-in for EmbeddingSimilarity. Real embedder pulls
    sentence-transformers (which pulls pandas, which the test conftest
    stubs to a non-functional shim — so the real path can't run here).
    The fake returns deterministic vectors per text, with an optional
    explicit mapping so tests can pin expected similarity outcomes
    without depending on the choice of model."""

    def __init__(self, mapping: dict | None = None, dim: int = 384):
        import numpy as np

        self._np = np
        self._mapping = mapping or {}
        self.dim = dim
        self.calls: list[list[str]] = []

    def _vec_from_text(self, text: str):
        np = self._np
        if text in self._mapping:
            return np.array(self._mapping[text], dtype=np.float32)
        seed = abs(hash(text)) % (2**32)
        rng = np.random.RandomState(seed)
        v = rng.randn(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def get_embeddings_batch(self, texts):
        np = self._np
        self.calls.append(list(texts))
        return np.array([self._vec_from_text(t) for t in texts], dtype=np.float32)


def _patch_embedder(monkeypatch, fake: _FakeEmbedder):
    """Wire the fake into the import path entity_ledger_service uses."""
    from graphora_server.services.entity_resolution import embedding_similarity

    monkeypatch.setattr(
        embedding_similarity,
        "get_embedding_similarity",
        lambda *args, **kwargs: fake,
    )


@pytest.mark.asyncio
async def test_record_nodes_persists_embedding_when_enabled(
    monkeypatch,
):
    """When ENTITY_RESOLUTION_EMBEDDING_ENABLED is on, record_nodes
    should compute and store an embedding + the active model name
    on the in-memory entry. This is the write side of slice 1 —
    without it the read-side (find_similar_entities) has nothing
    to consume."""
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(
        settings, "ENTITY_RESOLUTION_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )
    fake = _FakeEmbedder()
    _patch_embedder(monkeypatch, fake)

    service = EntityLedgerService(memory_store={})
    node = _node_with_canonical("Acme", "Company:name=acme")

    await service.record_nodes("user-1", [node])

    key = ("user-1", "Company", "Company:name=acme")
    entry = service._memory_store[key]
    assert entry.embedding is not None, (
        "record_nodes did not populate entry.embedding — slice 1 storage "
        "side is broken; find_similar_entities will see nothing."
    )
    assert isinstance(entry.embedding, list)
    assert len(entry.embedding) == fake.dim
    assert entry.embedding_model == "all-MiniLM-L6-v2"
    # Confirm the embedder was actually invoked for the right text.
    assert any("acme" in (t or "").lower() for t in fake.calls[0])


@pytest.mark.asyncio
async def test_record_nodes_skips_embedding_when_disabled(monkeypatch):
    """The embedding compute is the slowest part of record_nodes
    (loads sentence-transformers on first call). When the operator
    disables it, we should skip the work entirely — pin that the
    feature flag actually short-circuits."""
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_EMBEDDING_ENABLED", False)

    service = EntityLedgerService(memory_store={})
    node = _node_with_canonical("Acme", "Company:name=acme")

    await service.record_nodes("user-1", [node])

    entry = service._memory_store[("user-1", "Company", "Company:name=acme")]
    assert entry.embedding is None
    assert entry.embedding_model is None


@pytest.mark.asyncio
async def test_find_similar_entities_uses_stored_embedding(monkeypatch):
    """End-to-end: record a node (which embeds + stores), then ask
    find_similar_entities to look up a near-identical node by
    semantic similarity. Pre-slice-1 this would have re-embedded
    the stored entry every call; post-slice-1 the stored embedding
    is reused so only the query-side embedding is computed.

    The fake embedder pins identical vectors for the two texts so
    the assertion is on the *plumbing* — that the stored vector
    survives the round-trip and gets used in the similarity step.
    The real embedder's choice of representation isn't exercised
    here (deferred to integration tests once pandas isn't stubbed)."""
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(
        settings, "ENTITY_RESOLUTION_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    # The two text representations the service produces for these
    # nodes (via _node_to_text). Pin both to the same unit vector
    # so cosine similarity = 1.0 — this isolates the test from the
    # real embedder's behaviour and verifies only the plumbing.
    import numpy as np

    same_vec = np.zeros(384, dtype=np.float32)
    same_vec[0] = 1.0
    # _node_to_text emits only the canonical-properties values when the
    # regular property keys overlap (which they do here). Both nodes
    # collapse to a single token so the mapping keys stay simple.
    fake = _FakeEmbedder(
        mapping={
            "acme inc": same_vec.tolist(),
            "acme incorporated": same_vec.tolist(),
        }
    )
    _patch_embedder(monkeypatch, fake)

    service = EntityLedgerService(memory_store={})
    stored = _node_with_canonical("Acme Inc", "Company:name=acme-inc")
    await service.record_nodes("user-1", [stored])

    # Different canonical_key (so exact-match lookup misses) but
    # the fake embedder maps both to the same vector → match.
    query = _node_with_canonical("Acme Incorporated", "Company:name=acme-incorporated")

    matches = await service.find_similar_entities("user-1", [query], threshold=0.6)
    assert query.id in matches, (
        "find_similar_entities returned no match — either the stored "
        "embedding wasn't read back (slice 1 read path broken) or the "
        "similarity computation is using the wrong representation."
    )
    assert matches[query.id]["canonical_id"] == stored.canonical_id
    # Score must be > threshold; with identical mocked vectors it's 1.0
    # subject to floating-point clip.
    assert matches[query.id]["similarity"] >= 0.99


@pytest.mark.asyncio
async def test_find_similar_entities_skips_legacy_unembedded_entries(
    monkeypatch,
):
    """Pre-slice-1 ledger rows have embedding=NULL. The reader must
    skip them silently rather than crashing or treating NULL as a
    zero vector (which would always look perfectly similar)."""
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(
        settings, "ENTITY_RESOLUTION_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    service = EntityLedgerService(memory_store={})

    # Manually plant a legacy entry: no embedding, no embedding_model.
    from graphora_server.services.entity_ledger_service import EntityLedgerEntry

    legacy = EntityLedgerEntry(
        user_id="user-1",
        entity_type="Company",
        canonical_key="Company:name=legacy",
        canonical_id=_make_canonical_node_id("Company:name=legacy"),
        features={},
        confidence=1.0,
    )
    service._memory_store[("user-1", "Company", "Company:name=legacy")] = legacy

    query = _node_with_canonical("Anything", "Company:name=anything")
    matches = await service.find_similar_entities("user-1", [query], threshold=0.5)

    # No usable embeddings -> no matches; must NOT crash.
    assert matches == {}


@pytest.mark.asyncio
async def test_find_similar_entities_skips_model_mismatch(monkeypatch):
    """An entry embedded under a different model produces vectors
    in a different vector space; mixing them returns garbage scores.
    Pin that the reader skips them rather than letting them poison
    the similarity computation."""
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(
        settings, "ENTITY_RESOLUTION_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    service = EntityLedgerService(memory_store={})

    from graphora_server.services.entity_ledger_service import EntityLedgerEntry

    foreign = EntityLedgerEntry(
        user_id="user-1",
        entity_type="Company",
        canonical_key="Company:name=foreign",
        canonical_id=_make_canonical_node_id("Company:name=foreign"),
        features={},
        confidence=1.0,
        # Plausible-shaped vector but produced by a different model.
        embedding=[0.1] * 768,
        embedding_model="all-mpnet-base-v2",
    )
    service._memory_store[("user-1", "Company", "Company:name=foreign")] = foreign

    query = _node_with_canonical("Foreign", "Company:name=foreign-query")
    matches = await service.find_similar_entities("user-1", [query], threshold=0.5)

    assert matches == {}


@pytest.mark.asyncio
async def test_find_similar_entities_skips_unknown_model_with_embedding(
    monkeypatch,
):
    """An entry that has an embedding but no recorded model is the
    ambiguous case: we can't tell whether the vector is from the
    active model or a different one. Without skipping it we'd
    either silently mix vector spaces (garbage scores) or crash on
    a dimension mismatch in np.dot. Pin the conservative skip — see
    reviewer P2 finding on entity_ledger_service.py:389."""
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(
        settings, "ENTITY_RESOLUTION_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    service = EntityLedgerService(memory_store={})

    from graphora_server.services.entity_ledger_service import EntityLedgerEntry

    # Embedding present, model column NULL — possible after the
    # migration runs against rows written by an even-older code
    # path that populated embedding via some other means.
    legacy_with_vec = EntityLedgerEntry(
        user_id="user-1",
        entity_type="Company",
        canonical_key="Company:name=legacy-vec",
        canonical_id=_make_canonical_node_id("Company:name=legacy-vec"),
        features={},
        confidence=1.0,
        # Wrong dimension on purpose — if the skip fails, the dot
        # product will crash and the test surfaces that loud.
        embedding=[0.1] * 100,
        embedding_model=None,
    )
    service._memory_store[("user-1", "Company", "Company:name=legacy-vec")] = (
        legacy_with_vec
    )

    fake = _FakeEmbedder()
    _patch_embedder(monkeypatch, fake)
    query = _node_with_canonical("Query", "Company:name=query")

    # Must not crash on dimension mismatch; must return no match.
    matches = await service.find_similar_entities("user-1", [query], threshold=0.5)
    assert matches == {}


@pytest.mark.asyncio
async def test_get_entries_by_type_orders_and_caps_at_10k(monkeypatch):
    """Slice 1 made the per-type fetch the persisted similarity
    index. The pre-fix query had LIMIT 1000 with no ORDER BY, so
    rows beyond the first 1000 the planner returned were silently
    dropped and which 1000 won varied across reads. Pin the new
    contract: ORDER BY updated_at DESC + LIMIT 10000 — see reviewer
    P2 finding on entity_ledger_service.py:449."""
    captured_query: list[str] = []

    class _StubDb:
        async def fetch(self, query, *params):
            captured_query.append(query)
            return []

        async def executemany(self, query, params):
            pass

    from graphora_server.services import entity_ledger_service as ledger_module

    monkeypatch.setattr(ledger_module, "db", _StubDb())

    service = EntityLedgerService(memory_store={})
    # Force the DB-enabled path even though the autouse fixture
    # disables DATABASE_URL — the stub above doesn't care about
    # connection strings.
    service._enabled = True

    await service._get_entries_by_type("user-1", "Company")
    assert captured_query, "_get_entries_by_type didn't issue a fetch"
    sql = captured_query[0]
    assert "ORDER BY updated_at DESC" in sql, (
        "Per-type fetch must order by updated_at DESC so the LIMIT is "
        "deterministic and biased toward fresh entries."
    )
    assert "LIMIT 10000" in sql, (
        "Per-type fetch cap was bumped from 1000 to 10000 to match the "
        "migration-13 docstring; bringing it back below 10k re-introduces "
        "the silent-drop regression."
    )


@pytest.mark.asyncio
async def test_hydrate_nodes_dispatches_to_similarity_when_flag_on(
    monkeypatch,
):
    """Slice 2: with ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED=True,
    hydrate_nodes routes to hydrate_nodes_with_similarity. The legacy
    exact-key-only path must NOT run when the flag is on. Pin the
    dispatch so a refactor can't silently lose the cross-document
    feature."""
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED", True)
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_SIMILARITY_THRESHOLD", 0.85)

    service = EntityLedgerService(memory_store={})

    seen: dict[str, object] = {}

    async def fake_with_similarity(user_id, nodes, similarity_threshold=0.85):
        seen["called"] = True
        seen["user_id"] = user_id
        seen["threshold"] = similarity_threshold
        seen["nodes"] = list(nodes)

    monkeypatch.setattr(service, "hydrate_nodes_with_similarity", fake_with_similarity)

    node = _node_with_canonical("Acme", "Company:name=acme")
    await service.hydrate_nodes("user-1", [node])

    assert seen.get("called") is True
    assert seen["user_id"] == "user-1"
    assert seen["threshold"] == 0.85
    assert len(seen["nodes"]) == 1


@pytest.mark.asyncio
async def test_hydrate_nodes_uses_legacy_path_when_flag_off(monkeypatch):
    """Mirror of the above: with the flag off, the similarity path
    must NOT run — operators who haven't opted in should see
    bit-identical behaviour to the pre-slice-2 codebase. Pin this
    so a future refactor can't accidentally make similarity the
    default path."""
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED", False)

    service = EntityLedgerService(memory_store={})

    called = {"value": False}

    async def fail_if_called(*args, **kwargs):
        called["value"] = True

    monkeypatch.setattr(service, "hydrate_nodes_with_similarity", fail_if_called)

    # Stash an exact-key match so we can verify the legacy path still runs.
    canonical_key = "Company:name=acme"
    canonical_id = _make_canonical_node_id(canonical_key)
    service._memory_store[("user-1", "Company", canonical_key)] = __import__(
        "graphora_server.services.entity_ledger_service",
        fromlist=["EntityLedgerEntry"],
    ).EntityLedgerEntry(
        user_id="user-1",
        entity_type="Company",
        canonical_key=canonical_key,
        canonical_id=canonical_id,
        features={},
        confidence=1.0,
    )

    fresh_node = BaseNode(
        id="local-fresh",
        type="Company",
        properties={"name": "Acme"},
        canonical_properties={"name": "acme"},
        canonical_key=canonical_key,
    )
    await service.hydrate_nodes("user-1", [fresh_node])

    assert called["value"] is False
    # Legacy exact-key match still landed on the fresh node.
    assert fresh_node.canonical_id == canonical_id


@pytest.mark.asyncio
async def test_hydrate_summary_log_emitted_with_match_counts(monkeypatch, caplog):
    """Telemetry contract: each cross-doc hydrate call must emit one
    INFO summary line with structured ledger_* fields naming the
    counts. Operators rely on this to spot weird match rates without
    enabling debug-level logging."""
    import logging

    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED", True)
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_SIMILARITY_THRESHOLD", 0.85)

    service = EntityLedgerService(memory_store={})

    canonical_key = "Company:name=acme"
    canonical_id = _make_canonical_node_id(canonical_key)
    from graphora_server.services.entity_ledger_service import EntityLedgerEntry

    service._memory_store[("user-1", "Company", canonical_key)] = EntityLedgerEntry(
        user_id="user-1",
        entity_type="Company",
        canonical_key=canonical_key,
        canonical_id=canonical_id,
        features={},
        confidence=1.0,
    )

    matched = BaseNode(
        id="m1",
        type="Company",
        properties={"name": "Acme"},
        canonical_key=canonical_key,
    )
    unmatched = BaseNode(
        id="u1",
        type="Company",
        properties={"name": "Other"},
        canonical_key="Company:name=other",
    )

    fake = _FakeEmbedder()
    _patch_embedder(monkeypatch, fake)

    with caplog.at_level(
        logging.INFO, logger="graphora_server.services.entity_ledger_service"
    ):
        await service.hydrate_nodes("user-1", [matched, unmatched])

    summary_records = [
        r for r in caplog.records if "Cross-doc hydrate" in r.getMessage()
    ]
    assert len(summary_records) == 1, (
        "expected exactly one INFO summary per hydrate call; "
        f"got {len(summary_records)}: {[r.getMessage() for r in summary_records]}"
    )
    record = summary_records[0]
    # Structured fields make the line greppable in log aggregators.
    assert getattr(record, "ledger_total", None) == 2
    assert getattr(record, "ledger_exact_matches", None) == 1
    # No similarity match because the unmatched node has no candidate
    # to match against (the only stored entry was already claimed by
    # the exact-key stage).
    assert getattr(record, "ledger_similarity_matches", None) == 0
    assert getattr(record, "ledger_unmatched", None) == 1


class _FailingEmbedder:
    """Stand-in for an embedder that imports cleanly but raises at
    runtime — mirrors the model-load-fails / GPU-OOM / corrupted-
    weights case the lazy-load path makes possible. Slice 2's
    catch-and-degrade contract on find_similar_entities is what
    keeps a transform from becoming a 500 when the operator opts
    into cross-doc linking on a misconfigured runtime."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def get_embeddings_batch(self, texts):
        raise self._exc


@pytest.mark.asyncio
async def test_find_similar_entities_returns_empty_on_runtime_failure(
    monkeypatch,
):
    """The ImportError branch was the only safety net pre-fix; this
    pins the post-fix runtime-error branch. Without this, a
    misconfigured model (HF unreachable, missing weights, CUDA OOM)
    propagates a RuntimeError up to the caller and the entire
    transform fails — exactly the worst failure mode for a feature
    that's supposed to be additive on top of legacy exact-key
    resolution. Pin: any runtime exception → return {}, log
    warning, no propagation."""
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(
        settings, "ENTITY_RESOLUTION_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    failing = _FailingEmbedder(RuntimeError("model load failed"))
    _patch_embedder(monkeypatch, failing)

    service = EntityLedgerService(memory_store={})

    # Plant a usable entry with a matched-model embedding so the
    # filter loop reaches the runtime call. Without this, the
    # function short-circuits at the no-usable-entries check and we
    # never exercise the failing get_embeddings_batch path.
    from graphora_server.services.entity_ledger_service import EntityLedgerEntry

    service._memory_store[("user-1", "Company", "Company:name=stored")] = (
        EntityLedgerEntry(
            user_id="user-1",
            entity_type="Company",
            canonical_key="Company:name=stored",
            canonical_id=_make_canonical_node_id("Company:name=stored"),
            features={},
            confidence=1.0,
            embedding=[0.5] * 384,
            embedding_model="all-MiniLM-L6-v2",
        )
    )

    query = _node_with_canonical("QueryCo", "Company:name=queryco")

    # Must NOT raise; must return empty matches.
    result = await service.find_similar_entities("user-1", [query], threshold=0.5)
    assert result == {}


@pytest.mark.asyncio
async def test_hydrate_with_similarity_preserves_stage1_on_runtime_failure(
    monkeypatch,
):
    """End-to-end of the degradation contract: when Stage 2
    (similarity) crashes, Stage 1 (exact-key) matches that already
    landed on node.canonical_id must survive. This is what makes
    'degrade to exact-key only' actually meaningful — without it
    we'd be wiping legitimately-resolved nodes whenever the
    embedding runtime sneezed."""
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_CROSS_DOCUMENT_ENABLED", True)
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_EMBEDDING_ENABLED", True)
    monkeypatch.setattr(
        settings, "ENTITY_RESOLUTION_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    service = EntityLedgerService(memory_store={})

    # Stored entry with both an exact key and an embedding under
    # the active model — so both stages have something to match.
    canonical_key = "Company:name=acme"
    canonical_id = _make_canonical_node_id(canonical_key)
    from graphora_server.services.entity_ledger_service import EntityLedgerEntry

    service._memory_store[("user-1", "Company", canonical_key)] = EntityLedgerEntry(
        user_id="user-1",
        entity_type="Company",
        canonical_key=canonical_key,
        canonical_id=canonical_id,
        features={},
        confidence=1.0,
        embedding=[0.5] * 384,
        embedding_model="all-MiniLM-L6-v2",
    )

    # Force the Stage 2 model call to fail.
    failing = _FailingEmbedder(RuntimeError("CUDA out of memory"))
    _patch_embedder(monkeypatch, failing)

    matched = BaseNode(
        id="m1",
        type="Company",
        properties={"name": "Acme"},
        canonical_key=canonical_key,
    )
    unmatched = BaseNode(
        id="u1",
        type="Company",
        properties={"name": "Other"},
        canonical_key="Company:name=other",
    )

    # Must NOT raise.
    await service.hydrate_nodes("user-1", [matched, unmatched])

    # Stage 1 hit: matched.canonical_id was overridden by the ledger.
    assert matched.canonical_id == canonical_id
    # Stage 2 failed: the unmatched node carries no canonical_id
    # (None or empty — it never had one to begin with). The point
    # is that the failure didn't blow up the call.
    assert not unmatched.canonical_id


@pytest.mark.asyncio
async def test_record_nodes_handles_embedder_import_failure(
    monkeypatch,
):
    """sentence-transformers is an optional dependency. If the
    operator hasn't installed it, the embedding step should fail
    quietly and the rest of record_nodes (the canonical-key upsert)
    should still complete — pin the graceful-degradation contract."""
    monkeypatch.setattr(settings, "ENTITY_RESOLUTION_EMBEDDING_ENABLED", True)

    # Force the embedder import to look missing without uninstalling
    # the real package. monkeypatching sys.modules makes the import
    # statement inside _populate_embeddings_inplace raise ImportError.
    import sys

    monkeypatch.setitem(
        sys.modules,
        "graphora_server.services.entity_resolution.embedding_similarity",
        None,
    )

    service = EntityLedgerService(memory_store={})
    node = _node_with_canonical("Acme", "Company:name=acme")

    # Must not raise.
    await service.record_nodes("user-1", [node])

    # Canonical-key upsert still happened.
    entry = service._memory_store[("user-1", "Company", "Company:name=acme")]
    assert entry.canonical_id == node.canonical_id
    # Embedding stayed None because the embedder couldn't load.
    assert entry.embedding is None
