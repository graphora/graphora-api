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
