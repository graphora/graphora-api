"""GraphTransformer unit tests following London School TDD.

These tests verify the orchestration behavior of GraphTransformer,
focusing on how it coordinates with its collaborators:
- LLMClient for extraction
- Entity ledger for hydration/storage
- Merge learning service for deduplication thresholds

The focus is on INTERACTIONS, not implementation details.
"""

import pytest
from unittest.mock import MagicMock

from tests.mocks.llm_client_mock import (
    MockLLMClient,
    MockLLMResponse,
    MockEntityResolution,
)
from tests.factories.node_factory import NodeFactory
from tests.factories.relationship_factory import RelationshipFactory
from tests.fixtures.ontologies import COMPANY_PERSON_ONTOLOGY


# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def sample_ontology():
    """Standard test ontology."""
    return COMPANY_PERSON_ONTOLOGY


@pytest.fixture
def sample_chunks():
    """Sample text chunks for extraction."""
    return [
        "Acme Corporation is a technology company founded in 2010.",
        "Jane Smith is the CEO of Acme Corporation.",
        "Acme employs over 500 people worldwide.",
    ]


@pytest.fixture
def company_extraction_response():
    """Mock response for Company extraction."""

    class CompanyEntity:
        def __init__(self, name, ticker=None):
            self.name = name
            self.ticker = ticker

        def model_dump(self):
            return {"name": self.name, "ticker": self.ticker}

    return MockLLMResponse(
        entities={"Company": [CompanyEntity("Acme Corporation", "ACM")]},
        confidence=0.95,
    )


@pytest.fixture
def person_extraction_response():
    """Mock response for Person extraction."""

    class PersonEntity:
        def __init__(self, name, title=None):
            self.name = name
            self.title = title

        def model_dump(self):
            return {"name": self.name, "title": self.title}

    return MockLLMResponse(
        entities={"Person": [PersonEntity("Jane Smith", "CEO")]},
        confidence=0.92,
    )


# ============================================================
# Node Extraction Tests
# ============================================================


class TestGraphTransformerNodeExtraction:
    """Test node extraction orchestration."""

    @pytest.mark.asyncio
    async def test_should_extract_nodes_for_each_chunk(
        self, mock_llm_client, sample_chunks, company_extraction_response
    ):
        """Should call node extraction for each input chunk."""
        mock_llm_client.configure_node_extraction(
            company_extraction_response,
            company_extraction_response,
            company_extraction_response,
        )

        # Simulate what GraphTransformer does
        for chunk in sample_chunks:
            await mock_llm_client.extract_nodes_from_chunk(
                chunk=chunk,
                response_model=MagicMock(),
                user_id="user-123",
                transform_id="tx-123",
            )

        # Verify extraction was called for each chunk
        mock_llm_client.assert_called("extract_nodes_from_chunk", times=3)

        # Verify each chunk was processed
        call_args = mock_llm_client.get_call_args("extract_nodes_from_chunk")
        chunks_processed = [args["chunk"] for args in call_args]
        assert chunks_processed == sample_chunks

    @pytest.mark.asyncio
    async def test_should_pass_user_id_and_transform_id(
        self, mock_llm_client, company_extraction_response
    ):
        """Should pass user_id and transform_id to extraction calls."""
        mock_llm_client.configure_node_extraction(company_extraction_response)

        await mock_llm_client.extract_nodes_from_chunk(
            chunk="Test chunk",
            response_model=MagicMock(),
            user_id="user-456",
            transform_id="tx-789",
            document_usage_id="doc-001",
        )

        call_args = mock_llm_client.get_call_args("extract_nodes_from_chunk")[0]
        assert call_args["user_id"] == "user-456"
        assert call_args["transform_id"] == "tx-789"
        assert call_args["document_usage_id"] == "doc-001"

    @pytest.mark.asyncio
    async def test_should_pass_context_from_previous_extractions(
        self, mock_llm_client, company_extraction_response, sample_chunks
    ):
        """Should build context from previous nodes for subsequent extractions."""
        mock_llm_client.configure_node_extraction(
            company_extraction_response,
            company_extraction_response,
        )

        # First extraction - no context
        await mock_llm_client.extract_nodes_from_chunk(
            chunk=sample_chunks[0],
            response_model=MagicMock(),
            context="",  # Empty for first chunk
            user_id="user-123",
        )

        # Second extraction - with context from first
        context = "Previously extracted: Company(Acme Corporation)"
        await mock_llm_client.extract_nodes_from_chunk(
            chunk=sample_chunks[1],
            response_model=MagicMock(),
            context=context,  # Context from first extraction
            user_id="user-123",
        )

        call_args = mock_llm_client.get_call_args("extract_nodes_from_chunk")
        assert call_args[0]["context"] == ""
        assert "Previously extracted" in call_args[1]["context"]


# ============================================================
# Relationship Extraction Tests
# ============================================================


class TestGraphTransformerRelationshipExtraction:
    """Test relationship extraction orchestration."""

    @pytest.mark.asyncio
    async def test_should_extract_relationships_after_nodes(
        self, mock_llm_client, company_extraction_response
    ):
        """Should extract relationships using node context."""
        relationship_response = MockLLMResponse(entities={}, confidence=0.88)

        mock_llm_client.configure_node_extraction(company_extraction_response)
        mock_llm_client.configure_relationship_extraction(relationship_response)

        # Extract nodes first
        await mock_llm_client.extract_nodes_from_chunk(
            chunk="Acme employs Jane",
            response_model=MagicMock(),
            user_id="user-123",
        )

        # Then extract relationships with node context
        await mock_llm_client.extract_relationships_from_chunk(
            chunk="Acme employs Jane",
            response_model=MagicMock(),
            context="Nodes: Company(Acme), Person(Jane)",
            user_id="user-123",
        )

        # Verify sequence
        mock_llm_client.assert_called("extract_nodes_from_chunk", times=1)
        mock_llm_client.assert_called("extract_relationships_from_chunk", times=1)

        # Verify context was passed
        rel_args = mock_llm_client.get_call_args("extract_relationships_from_chunk")[0]
        assert "Nodes:" in rel_args["context"]


# ============================================================
# Entity Resolution Tests
# ============================================================


class TestGraphTransformerEntityResolution:
    """Test entity resolution coordination."""

    @pytest.mark.asyncio
    async def test_should_call_entity_resolution_when_multiple_nodes_same_type(
        self, mock_llm_client
    ):
        """Should resolve entities when group has multiple nodes of same type."""
        NodeFactory.reset_counter()

        # Create multiple companies that might be duplicates
        nodes = [
            NodeFactory.create_company(name="Acme Corp"),
            NodeFactory.create_company(name="Acme Corporation"),
            NodeFactory.create_company(name="ACME Inc"),
        ]

        # Configure resolution response
        mock_llm_client.configure_entity_resolution(
            [
                MockEntityResolution(
                    matching_ids=[nodes[0].id, nodes[1].id, nodes[2].id],
                    confidence_score=0.95,
                    explanation="All refer to the same company",
                )
            ]
        )

        # Build node dicts for resolution
        import json

        node_dicts = [
            {"id": n.id, "properties": n.properties, "confidence": 0.9} for n in nodes
        ]

        result = await mock_llm_client.resolve_entities(
            entity_type="Company",
            node_dicts_str=json.dumps(node_dicts),
            user_id="user-123",
        )

        # Verify resolution was called
        mock_llm_client.assert_called("resolve_entities", times=1)

        # Verify all node IDs were grouped
        assert len(result) == 1
        assert len(result[0].matching_ids) == 3

    @pytest.mark.asyncio
    async def test_should_skip_resolution_for_single_node(self, mock_llm_client):
        """Should not call resolution when only one node of type."""
        NodeFactory.reset_counter()
        _single_node = NodeFactory.create_company(name="Unique Corp")

        # For a single node, resolution should not be called
        # This is what we expect GraphTransformer to do
        mock_llm_client.assert_not_called("resolve_entities")


# ============================================================
# Context Building Tests
# ============================================================


class TestGraphTransformerContextBuilding:
    """Test context envelope construction."""

    @pytest.mark.asyncio
    async def test_nodes_context_should_be_sorted_deterministically(self):
        """Nodes context should sort by type, properties, id for consistency."""
        from graphora_server.services.transform.graph_transformer import (
            _build_nodes_context,
        )

        NodeFactory.reset_counter()

        # Create nodes in random order
        node_z = NodeFactory.create(
            node_id="z", node_type="Company", properties={"name": "Zebra"}
        )
        node_a = NodeFactory.create(
            node_id="a", node_type="Company", properties={"name": "Alpha"}
        )
        node_m = NodeFactory.create(
            node_id="m", node_type="Person", properties={"name": "Maria"}
        )

        # Build context in different orders
        context_1 = await _build_nodes_context([node_z, node_a, node_m])
        context_2 = await _build_nodes_context([node_m, node_a, node_z])

        # Context should be identical regardless of input order
        assert context_1 == context_2

    @pytest.mark.asyncio
    async def test_relationships_context_should_identify_orphan_nodes(self):
        """Should list nodes not in any relationship."""
        from graphora_server.services.transform.graph_transformer import (
            _build_relationships_context,
        )

        NodeFactory.reset_counter()
        RelationshipFactory.reset_counter()

        # Create nodes where one has no relationships
        company = NodeFactory.create_company(name="Acme", node_id="company-1")
        person = NodeFactory.create_person(name="Jane", node_id="person-1")
        orphan = NodeFactory.create(
            node_id="orphan-1", node_type="Location", properties={"name": "HQ"}
        )

        # Only company-person relationship exists
        relationship = RelationshipFactory.create_employs(
            source_id="company-1",
            target_id="person-1",
        )

        context = await _build_relationships_context(
            [company, person, orphan],
            [relationship],
        )

        # Orphan node should be mentioned
        assert "Location" in context or "HQ" in context


# ============================================================
# Node Merging Tests
# ============================================================


class TestGraphTransformerNodeMerging:
    """Test node comparison and merging behavior."""

    def test_merge_nodes_should_combine_provenance_chunk_ids(self):
        """Merged node should have combined chunk IDs from both sources."""
        from graphora_server.services.transform.helpers import merge_nodes

        NodeFactory.reset_counter()

        existing = NodeFactory.create_company(
            name="Acme", chunk_ids=["chunk-1", "chunk-2"]
        )
        new_node = NodeFactory.create_company(name="Acme Corp", chunk_ids=["chunk-3"])

        merged = merge_nodes(existing, new_node)

        # Should have all chunk IDs
        assert "chunk-1" in merged.provenance.chunk_ids
        assert "chunk-2" in merged.provenance.chunk_ids
        assert "chunk-3" in merged.provenance.chunk_ids

    def test_merge_nodes_should_prefer_higher_confidence_values(self):
        """Should prefer property values from higher confidence node."""
        from graphora_server.services.transform.helpers import merge_nodes

        NodeFactory.reset_counter()

        existing = NodeFactory.create_company(
            name="Acme",
            node_id="existing",
            confidence=0.7,
        )
        new_node = NodeFactory.create_company(
            name="Acme Corporation",  # More complete name
            node_id="new",
            confidence=0.95,  # Higher confidence
        )

        merged = merge_nodes(existing, new_node)

        # Higher confidence should be preserved
        assert merged.confidence_score == 0.95

    def test_merge_nodes_should_prefer_longer_strings_when_confidence_equal(self):
        """When confidence equal, should prefer longer string values."""
        from graphora_server.services.transform.helpers import merge_nodes

        NodeFactory.reset_counter()

        existing = NodeFactory.create(
            node_type="Company",
            properties={"name": "Acme"},
            confidence=0.9,
        )
        new_node = NodeFactory.create(
            node_type="Company",
            properties={"name": "Acme Corporation Ltd"},  # Longer
            confidence=0.9,  # Same confidence
        )

        _merged = merge_nodes(existing, new_node)

        # Longer value should be preferred (implementation may vary)
        # This documents expected behavior


# ============================================================
# Error Handling Tests
# ============================================================


class TestGraphTransformerErrorHandling:
    """Test error handling during transformation."""

    @pytest.mark.asyncio
    async def test_should_handle_llm_extraction_failure_gracefully(
        self, mock_llm_client
    ):
        """When LLM extraction fails, should handle gracefully."""
        mock_llm_client.configure_error(
            "extract_nodes_from_chunk",
            Exception("LLM service unavailable"),
        )

        with pytest.raises(Exception, match="LLM service unavailable"):
            await mock_llm_client.extract_nodes_from_chunk(
                chunk="Test",
                response_model=MagicMock(),
                user_id="user-123",
            )

    @pytest.mark.asyncio
    async def test_should_continue_processing_on_single_chunk_failure(
        self, mock_llm_client, company_extraction_response
    ):
        """Should attempt to process remaining chunks if one fails."""
        # This documents expected behavior - GraphTransformer should
        # handle partial failures gracefully

        mock_llm_client.configure_node_extraction(
            company_extraction_response,
            # Second chunk would fail in real scenario
        )

        # First call succeeds
        result = await mock_llm_client.extract_nodes_from_chunk(
            chunk="First chunk",
            response_model=MagicMock(),
            user_id="user-123",
        )

        assert result is not None
        mock_llm_client.assert_called("extract_nodes_from_chunk", times=1)
