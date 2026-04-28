"""Unit tests for build_full_text_indexes_for_user backend dispatch.

Slice 8 round-2 review: the live ontology index-creation path used
to hard-wire Neo4jStorage and ignore STORAGE_TYPE=postgres. These
tests pin that the dispatch now routes through the configured
backend so the AGE adapter's GIN polyfill actually runs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphora_server.services.transform.ontology_helper import OntologyParser


def _make_parser_with_ontology() -> OntologyParser:
    """Construct an OntologyParser with a minimal in-memory ontology
    so the loop has at least one entity + relationship to walk.
    Avoids touching disk or the Supabase fallback path."""
    parser = OntologyParser.__new__(OntologyParser)
    parser.ontology_yaml = ""
    parser.parsed_ontology = {
        "entities": {
            "Person": {
                "properties": {"name": {"type": "string"}},
                "relationships": {
                    "WORKS_AT": {
                        "target": "Company",
                        "properties": {"role": {"type": "string"}},
                    }
                },
            },
            "Company": {"properties": {"name": {"type": "string"}}},
        }
    }
    return parser


@pytest.mark.asyncio
async def test_memory_mode_skips_index_creation(monkeypatch):
    """STORAGE_TYPE=memory: no FT indexes, no storage construction,
    no UserDatabaseService lookup."""
    monkeypatch.setattr("graphora_server.config.settings.STORAGE_TYPE", "memory")
    parser = _make_parser_with_ontology()

    # Sentinel patches — must NOT be called.
    udb_get = AsyncMock()
    monkeypatch.setattr(
        "graphora_server.services.user_db_service.UserDatabaseService.get_user_config",
        udb_get,
    )

    await parser.build_full_text_indexes_for_user("u1")
    udb_get.assert_not_called()


@pytest.mark.asyncio
async def test_postgres_mode_dispatches_to_age_adapter(monkeypatch):
    """STORAGE_TYPE=postgres: index creation must go through
    _build_age_storage (the factory shared helper) and call the
    AGE adapter's GIN polyfill, not Neo4jStorage."""
    monkeypatch.setattr("graphora_server.config.settings.STORAGE_TYPE", "postgres")

    fake_age = MagicMock()
    fake_age.get_all_node_properties = AsyncMock(return_value=[])
    fake_age.get_all_relationship_properties = AsyncMock(return_value=[])
    fake_age.create_or_replace_ft_index_for_node = AsyncMock()
    fake_age.create_or_replace_ft_index_for_relationship = AsyncMock()

    monkeypatch.setattr(
        "graphora_server.services.storage.factory._build_age_storage",
        lambda: fake_age,
    )
    # If the old code path runs, Neo4jStorage construction would
    # fail — but the import is local to the neo4j branch, so a
    # missing-config error surface is good enough as a smoke
    # signal.

    parser = _make_parser_with_ontology()
    await parser.build_full_text_indexes_for_user("u1")

    # The AGE adapter received node-index calls (one per entity
    # type with properties). Both staging and prod point at the
    # same shared instance, so each entity gets two calls.
    assert fake_age.create_or_replace_ft_index_for_node.call_count >= 1
    assert fake_age.create_or_replace_ft_index_for_relationship.call_count >= 1


@pytest.mark.asyncio
async def test_neo4j_mode_preserves_per_user_construction(monkeypatch):
    """STORAGE_TYPE=neo4j: existing per-user staging+prod path
    survives the refactor unchanged."""
    monkeypatch.setattr("graphora_server.config.settings.STORAGE_TYPE", "neo4j")

    user_config = MagicMock()
    user_config.stagingDb = MagicMock(
        uri="bolt://staging:7687", username="u", password="p"
    )
    user_config.prodDb = None  # only staging configured

    monkeypatch.setattr(
        "graphora_server.services.user_db_service.UserDatabaseService.get_user_config",
        AsyncMock(return_value=user_config),
    )

    fake_neo4j = MagicMock()
    fake_neo4j.get_all_node_properties = AsyncMock(return_value=[])
    fake_neo4j.get_all_relationship_properties = AsyncMock(return_value=[])
    fake_neo4j.create_or_replace_ft_index_for_node = AsyncMock()
    fake_neo4j.create_or_replace_ft_index_for_relationship = AsyncMock()

    # Patch the Neo4jStorage class itself — the function imports
    # it locally inside the branch.
    monkeypatch.setattr(
        "graphora_server.services.storage.neo4j.Neo4jStorage",
        lambda **kwargs: fake_neo4j,
    )

    parser = _make_parser_with_ontology()
    await parser.build_full_text_indexes_for_user("u1")

    # Staging path called; prod path skipped (prodDb None).
    assert fake_neo4j.create_or_replace_ft_index_for_node.called
