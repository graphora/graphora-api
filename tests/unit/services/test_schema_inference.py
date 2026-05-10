"""Unit tests for schema inference service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graphora_server.services.schema_inference import (
    infer_schema_from_text,
    _parse_yaml_response,
    create_auto_schema_ontology,
    get_default_generic_schema,
)


class TestParseYamlResponse:
    """Tests for YAML response parsing."""

    def test_parse_plain_yaml(self):
        """Test parsing plain YAML without code blocks."""
        yaml_text = """version: "0.1.0"
entities:
  Person:
    description: "A person"
    properties:
      name:
        type: str
        required: true
relationships: {}
"""
        result = _parse_yaml_response(yaml_text)
        assert result["version"] == "0.1.0"
        assert "Person" in result["entities"]
        assert result["entities"]["Person"]["properties"]["name"]["type"] == "str"

    def test_parse_yaml_with_code_block(self):
        """Test parsing YAML from markdown code block."""
        yaml_text = """Here's the schema:

```yaml
version: "0.1.0"
entities:
  Company:
    description: "A company"
    properties:
      name:
        type: str
```

That's the generated schema."""
        result = _parse_yaml_response(yaml_text)
        assert result["version"] == "0.1.0"
        assert "Company" in result["entities"]

    def test_parse_yaml_with_generic_code_block(self):
        """Test parsing YAML from generic code block without yaml tag."""
        yaml_text = """```
version: "0.1.0"
entities:
  Product:
    properties:
      name:
        type: str
```"""
        result = _parse_yaml_response(yaml_text)
        assert "Product" in result["entities"]

    def test_parse_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises ValueError."""
        invalid_yaml = "this: is: not: valid: yaml: [["
        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_yaml_response(invalid_yaml)

    def test_parse_non_dict_yaml_raises_error(self):
        """Test that non-dict YAML raises ValueError."""
        list_yaml = "- item1\n- item2"
        with pytest.raises(ValueError, match="Expected dict"):
            _parse_yaml_response(list_yaml)


class TestGetDefaultGenericSchema:
    """Tests for default generic schema."""

    def test_default_schema_is_valid_yaml(self):
        """Test that default schema is valid YAML."""
        import yaml

        schema_yaml = get_default_generic_schema()
        schema = yaml.safe_load(schema_yaml)

        assert "version" in schema
        assert "entities" in schema
        assert "relationships" in schema

    def test_default_schema_has_basic_entities(self):
        """Test that default schema has common entity types."""
        import yaml

        schema_yaml = get_default_generic_schema()
        schema = yaml.safe_load(schema_yaml)

        assert "Person" in schema["entities"]
        assert "Organization" in schema["entities"]
        assert "Concept" in schema["entities"]

    def test_default_schema_has_relationships(self):
        """Test that default schema has common relationships."""
        import yaml

        schema_yaml = get_default_generic_schema()
        schema = yaml.safe_load(schema_yaml)

        assert "RELATED_TO" in schema["relationships"]
        assert "WORKS_AT" in schema["relationships"]


class TestInferSchemaFromText:
    """Tests for schema inference from text."""

    @pytest.mark.asyncio
    async def test_infer_schema_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="No text content"):
            await infer_schema_from_text([], "test-user")

    @pytest.mark.asyncio
    async def test_infer_schema_whitespace_only_raises_error(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="No text content"):
            await infer_schema_from_text(["   ", "\n\n"], "test-user")

    @pytest.mark.asyncio
    async def test_infer_schema_calls_llm(self):
        """Test that schema inference calls the LLM correctly."""
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    description: "A person"
    properties:
      name:
        type: str
        required: true
relationships: {}
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "graphora_server.services.schema_inference.get_llm_client_for_user",
            new_callable=AsyncMock,
            return_value=(mock_client, "model-name", "gemini"),
        ):
            result = await infer_schema_from_text(
                ["Sample text about people and companies."],
                "test-user",
            )

            assert "entities" in result
            assert "Person" in result["entities"]
            mock_client.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_infer_schema_adds_missing_version(self):
        """Test that missing version is added."""
        mock_response = MagicMock()
        mock_response.text = """entities:
  Person:
    properties:
      name:
        type: str
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "graphora_server.services.schema_inference.get_llm_client_for_user",
            new_callable=AsyncMock,
            return_value=(mock_client, "model-name", "gemini"),
        ):
            result = await infer_schema_from_text(["Sample text"], "test-user")

            assert result["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_infer_schema_adds_missing_relationships(self):
        """Test that missing relationships dict is added."""
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: str
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "graphora_server.services.schema_inference.get_llm_client_for_user",
            new_callable=AsyncMock,
            return_value=(mock_client, "model-name", "gemini"),
        ):
            result = await infer_schema_from_text(["Sample text"], "test-user")

            assert "relationships" in result
            assert isinstance(result["relationships"], dict)


class TestCreateAutoSchemaOntology:
    """Tests for creating auto-schema ontology."""

    @pytest.fixture(autouse=True)
    def _force_memory_mode(self, monkeypatch):
        """B0-log slice 3a (commit 52995d4) made
        ``create_auto_schema_ontology`` unconditionally construct a
        ``DecisionLogService``. ``tests/conftest.py`` sets a default
        ``DATABASE_URL`` for tests that need it, which flips the
        service into Postgres mode and opens a psycopg pool on the
        first append. The pool open is slow (seconds) and the
        append fails-and-swallows against the unreachable DB.

        Existing tests in this class don't care about the Decision
        Log behaviour — they were written before slice 3a — but
        without this autouse fixture they'd silently pay the
        pool-open cost on every run. Force memory mode at the
        class boundary so all tests here read decisions in-process
        (or, for tests that don't read them at all, just no-op
        cheaply)."""
        from graphora_server.config import settings

        monkeypatch.setattr(settings, "DATABASE_URL", "")
        monkeypatch.setattr(settings, "POSTGRES_HOST", None)

    @pytest.mark.asyncio
    async def test_create_ontology_stores_in_database(self):
        """Test that auto-schema ontology is stored."""
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: str
relationships: {}
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "graphora_server.services.schema_inference.get_llm_client_for_user",
            new_callable=AsyncMock,
            return_value=(mock_client, "model-name", "gemini"),
        ):
            with patch(
                "graphora_server.services.schema_inference.ontology_storage_service.store_ontology",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_store:
                ontology_id = await create_auto_schema_ontology(
                    text_chunks=["Sample text"],
                    user_id="test-user",
                    transform_id="transform_abc123",
                )

                assert ontology_id.startswith("auto_")
                mock_store.assert_called_once()
                call_args = mock_store.call_args
                assert call_args.kwargs["user_id"] == "test-user"
                assert "Auto-generated" in call_args.kwargs["name"]

    @pytest.mark.asyncio
    async def test_infer_schema_records_llm_usage_under_schema_inference_op(
        self,
    ):
        """Reviewer-flagged on commit 34e29d7 (B5-obs slice 1): the
        Gemini call inside infer_schema_from_text wasn't wrapped by
        usage tracking, so the /cost endpoint's
        ``by_operation_type`` dropped the schema_inference bucket
        for the default auto-schema path (the most common transform
        flow).

        Pin: when transform_id is supplied, the call invokes
        track_llm_completion with operation_type='schema_inference',
        the matching transform_id, AND the resolved provider
        (Gemini in this case)."""
        from graphora_server.schemas.usage import ModelProvider

        # A response object the LLM-usage tracker can pull
        # tokens off via ``set_usage_from_response``.
        usage_metadata = MagicMock()
        usage_metadata.prompt_token_count = 500
        usage_metadata.candidates_token_count = 100
        usage_metadata.total_token_count = 600
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: str
relationships: {}
"""
        mock_response.usage_metadata = usage_metadata
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with (
            patch(
                "graphora_server.services.schema_inference.get_llm_client_for_user",
                new_callable=AsyncMock,
                return_value=(mock_client, "gemini-2.5-flash", "gemini"),
            ),
            patch(
                "graphora_server.services.schema_inference.track_llm_completion",
                new_callable=AsyncMock,
            ) as mock_track,
        ):
            await infer_schema_from_text(
                text_chunks=["Alice joined Acme."],
                user_id="user-1",
                transform_id="tx-auto-schema",
            )

        assert mock_track.await_count == 1
        kwargs = mock_track.await_args.kwargs
        assert kwargs["user_id"] == "user-1"
        assert kwargs["transform_id"] == "tx-auto-schema"
        # The bucket key the cost report partitions by — must
        # match exactly so by_operation_type lights up.
        assert kwargs["operation_type"] == "schema_inference"
        assert kwargs["model_name"] == "gemini-2.5-flash"
        # Provider matches what get_llm_client_for_user returned.
        assert kwargs["model_provider"] == ModelProvider.GEMINI
        # The response is forwarded so the tracker can extract
        # usage_metadata.
        assert kwargs["response"] is mock_response

    @pytest.mark.asyncio
    async def test_infer_schema_records_ollama_provider_truthfully(self):
        """Reviewer-flagged on commit 9f8e5e7 (B5-obs slice 1
        follow-up): ``get_llm_client_for_user`` can return
        provider='ollama', but the previous fix hard-coded
        ModelProvider.GEMINI inside track_gemini_usage, so /cost
        recorded ``gemini:<ollama_model>`` with zero tokens — the
        Gemini path was fixed but the provider-aware path produced
        misleading rows.

        Pin: when the LLM helper returns 'ollama', the tracking
        call passes ModelProvider.OLLAMA. The Ollama compat
        wrapper carries the model name through ``model_name``,
        so the cost report's ``models_used`` shows
        ``ollama:llama3.2`` rather than ``gemini:llama3.2``."""
        from graphora_server.schemas.usage import ModelProvider

        # Ollama compat response from llm_helper synthesizes a
        # Gemini-shaped usage_metadata so the same tracker
        # extractor picks up the tokens.
        usage_metadata = MagicMock()
        usage_metadata.prompt_token_count = 200
        usage_metadata.candidates_token_count = 50
        usage_metadata.total_token_count = 250
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: str
"""
        mock_response.usage_metadata = usage_metadata
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with (
            patch(
                "graphora_server.services.schema_inference.get_llm_client_for_user",
                new_callable=AsyncMock,
                return_value=(mock_client, "llama3.2", "ollama"),
            ),
            patch(
                "graphora_server.services.schema_inference.track_llm_completion",
                new_callable=AsyncMock,
            ) as mock_track,
        ):
            await infer_schema_from_text(
                text_chunks=["Alice joined Acme."],
                user_id="user-1",
                transform_id="tx-ollama",
            )

        assert mock_track.await_count == 1
        kwargs = mock_track.await_args.kwargs
        # The truthful provider — NOT GEMINI.
        assert kwargs["model_provider"] == ModelProvider.OLLAMA
        assert kwargs["model_name"] == "llama3.2"
        assert kwargs["operation_type"] == "schema_inference"

    @pytest.mark.asyncio
    async def test_infer_schema_skips_tracking_for_unknown_provider(self):
        """Defensive pin for the unknown-provider fallback:
        ``_resolve_model_provider`` returns None for any provider
        string not in the ModelProvider enum, and the call site
        skips tracking rather than mislabel the row. Operators
        will see the warning log and know to extend the enum."""
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: str
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with (
            patch(
                "graphora_server.services.schema_inference.get_llm_client_for_user",
                new_callable=AsyncMock,
                # Provider string the enum doesn't know about (yet).
                return_value=(mock_client, "model-x", "novel-provider"),
            ),
            patch(
                "graphora_server.services.schema_inference.track_llm_completion",
                new_callable=AsyncMock,
            ) as mock_track,
        ):
            await infer_schema_from_text(
                text_chunks=["Sample"],
                user_id="user-1",
                transform_id="tx-unknown",
            )

        # Tracking deliberately skipped — undercounting beats
        # mislabeling.
        mock_track.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_infer_schema_skips_tracking_when_transform_id_omitted(
        self,
    ):
        """Legacy callers (or any future call path) that don't
        supply transform_id should NOT invoke usage tracking. The
        log row would be unfindable in /cost (which keys on
        transform_id), and recording orphaned rows would skew
        per-user aggregates."""
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: str
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with (
            patch(
                "graphora_server.services.schema_inference.get_llm_client_for_user",
                new_callable=AsyncMock,
                return_value=(mock_client, "gemini-2.5-flash", "gemini"),
            ),
            patch(
                "graphora_server.services.schema_inference.track_llm_completion",
                new_callable=AsyncMock,
            ) as mock_track,
        ):
            await infer_schema_from_text(text_chunks=["Sample"], user_id="user-1")

        mock_track.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_auto_schema_threads_transform_id_into_inference(
        self,
    ):
        """Pin the threading: create_auto_schema_ontology must pass
        its transform_id into infer_schema_from_text so the
        downstream tracking call lands on the right transform.
        Without this thread, the tracking inside infer_schema would
        always be a no-op for the default auto-schema flow because
        transform_id would be None."""
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: str
relationships: {}
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with (
            patch(
                "graphora_server.services.schema_inference.get_llm_client_for_user",
                new_callable=AsyncMock,
                return_value=(mock_client, "gemini-2.5-flash", "gemini"),
            ),
            patch(
                "graphora_server.services.schema_inference.ontology_storage_service.store_ontology",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "graphora_server.services.schema_inference.track_llm_completion",
                new_callable=AsyncMock,
            ) as mock_track,
        ):
            await create_auto_schema_ontology(
                text_chunks=["Sample text"],
                user_id="user-1",
                transform_id="tx-auto-1",
            )

        # Tracking call ran, with the transform_id from the
        # outer create_auto_schema_ontology call. If a future
        # refactor drops the kwarg in the inference call, this
        # fails loud with transform_id=None.
        assert mock_track.await_count == 1
        assert mock_track.await_args.kwargs["transform_id"] == "tx-auto-1"
        assert mock_track.await_args.kwargs["operation_type"] == "schema_inference"

    @pytest.mark.asyncio
    async def test_create_ontology_raises_on_store_failure(self):
        """Test that failure to store raises ValueError."""
        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: str
relationships: {}
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "graphora_server.services.schema_inference.get_llm_client_for_user",
            new_callable=AsyncMock,
            return_value=(mock_client, "model-name", "gemini"),
        ):
            with patch(
                "graphora_server.services.schema_inference.ontology_storage_service.store_ontology",
                new_callable=AsyncMock,
                return_value=False,
            ):
                with pytest.raises(ValueError, match="Failed to store"):
                    await create_auto_schema_ontology(
                        text_chunks=["Sample text"],
                        user_id="test-user",
                        transform_id="transform_abc123",
                    )

    @pytest.mark.asyncio
    async def test_create_ontology_emits_schema_inferred_decision(self):
        """B0-log slice 3a: a successful auto-schema ontology
        creation emits one ``schema_inferred`` Decision so the
        Decision Log surface (Evidence tab, MCP get_evidence) can
        render "schema was auto-inferred from N chunks". Pin the
        contract directly: target_kind=SCHEMA, target_id=None,
        evidence carries the ontology_id + entity/relationship
        counts. Memory mode is forced at the class fixture so we
        can read appended rows directly without mocking psycopg."""
        from graphora_server.services.decision_log_service import (
            DecisionLogService,
            DecisionType,
            TargetKind,
        )

        # Patch DecisionLogService.__init__ side path: capture the
        # one instance the production code constructs so we can read
        # its memory_store. Patching the class itself with a hook
        # that records instances is the cleanest way without leaking
        # global state from the service.
        captured_logs: list = []

        class _CapturingLog(DecisionLogService):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured_logs.append(self)

        mock_response = MagicMock()
        mock_response.text = """version: "0.1.0"
entities:
  Person:
    properties:
      name:
        type: str
  Company:
    properties:
      name:
        type: str
relationships:
  WORKS_AT:
    source: Person
    target: Company
"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with (
            patch(
                "graphora_server.services.schema_inference.get_llm_client_for_user",
                new_callable=AsyncMock,
                return_value=(mock_client, "model-name", "gemini"),
            ),
            patch(
                "graphora_server.services.schema_inference.ontology_storage_service.store_ontology",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "graphora_server.services.schema_inference.DecisionLogService",
                new=_CapturingLog,
            ),
        ):
            ontology_id = await create_auto_schema_ontology(
                text_chunks=["Alice works at Acme.", "Bob works at Acme too."],
                user_id="user-1",
                transform_id="tx-auto-schema-1",
            )

        # One DecisionLogService constructed; one decision appended.
        assert len(captured_logs) == 1
        decisions = await captured_logs[0].for_transform("tx-auto-schema-1")
        assert len(decisions) == 1, (
            "Expected exactly 1 schema_inferred decision per "
            "create_auto_schema_ontology call. Got: "
            f"{[d.decision_type.value for d in decisions]}"
        )

        decision = decisions[0]
        assert decision.target_kind == TargetKind.SCHEMA
        assert decision.target_id is None
        assert decision.decision_type == DecisionType.SCHEMA_INFERRED

        # Evidence carries the inferred-schema metrics.
        ev = decision.evidence
        assert ev["ontology_id"] == ontology_id
        assert ev["text_chunk_count"] == 2
        # Two entity types in the inferred YAML, one relationship.
        assert ev["entities_count"] == 2
        assert ev["relationships_count"] == 1
        assert ev["version"] == "0.1.0"
