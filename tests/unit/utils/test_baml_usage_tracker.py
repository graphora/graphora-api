"""Unit tests for the BAMLUsageTracker — focused on the
effective_model_name plumbing (B5-obs slice 3 P2 fix).

Reviewer-flagged on commit 71923d4: model_override in the LLM
client routed refinement calls correctly, but the usage tracker
recorded ``call.client_name`` — the synthetic BAML alias
(``DynamicGemini``/``DynamicOllama``) set at registry construction.
That means primary ``gemini-1.5-flash`` and refinement
``gemini-2.5-pro`` calls would both land in llm_usage as the same
name, making /cost reports and models_used aggregations
misleading. These tests pin the fix: the LLMUsageRequest emitted
to usage_tracking_service carries the routed model name when the
caller threads it through.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphora_server.schemas.usage import ModelProvider
from graphora_server.utils.baml_usage_tracker import BAMLUsageTracker


def _fake_function_log(
    *,
    provider: str = "google-ai",
    client_name: str = "DynamicGemini",
    input_tokens: int = 100,
    output_tokens: int = 50,
    selected: bool = True,
) -> MagicMock:
    """Build a FunctionLog-shaped mock with one call. The shape
    mirrors what BAML's Collector.last returns — selected calls
    are the successful ones; client_name is the synthetic alias
    we want to OVERRIDE."""
    fake_call = MagicMock()
    fake_call.provider = provider
    fake_call.client_name = client_name
    fake_call.selected = selected
    fake_call.http_response = None

    fake_log = MagicMock()
    fake_log.calls = [fake_call]
    fake_log.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    fake_log.timing = MagicMock(duration_ms=42)
    return fake_log


class TestExtractModelInfo:
    """Direct exercise of BAMLUsageTracker._extract_model_info."""

    def test_effective_model_name_overrides_synthetic_alias(self) -> None:
        """Pin: when the caller provides effective_model_name, the
        synthetic BAML alias from call.client_name is replaced. The
        provider mapping (Gemini/OpenAI/etc.) is preserved — only
        the model name changes."""
        tracker = BAMLUsageTracker(
            user_id="u-1",
            operation_type="t",
            effective_model_name="gemini-2.5-pro",
        )
        log = _fake_function_log(provider="google-ai", client_name="DynamicGemini")
        provider, model_name = tracker._extract_model_info(log)
        assert provider == ModelProvider.GEMINI
        assert model_name == "gemini-2.5-pro", (
            "_extract_model_info ignored effective_model_name and "
            f"returned the synthetic alias: {model_name!r}"
        )

    def test_falls_back_to_client_name_when_no_override(self) -> None:
        """Back-compat: legacy call sites that don't supply
        effective_model_name keep their pre-fix behavior — the
        synthetic alias is recorded. Tracker isn't broken; it's
        just degraded for those paths."""
        tracker = BAMLUsageTracker(user_id="u-1", operation_type="t")
        log = _fake_function_log(provider="google-ai", client_name="DynamicGemini")
        _provider, model_name = tracker._extract_model_info(log)
        assert model_name == "DynamicGemini"

    def test_provider_mapping_routes_correctly(self) -> None:
        """Sanity: provider mapping logic from the function_log
        still works for each supported provider when an override
        is set. Pin so a future refactor of the mapping doesn't
        silently swap providers."""
        cases = [
            ("google-ai", ModelProvider.GEMINI),
            ("openai", ModelProvider.OPENAI),
            ("anthropic", ModelProvider.ANTHROPIC),
            ("openai-generic", ModelProvider.OPENAI),
            ("something-weird", ModelProvider.BAML),
        ]
        for provider_str, expected_enum in cases:
            tracker = BAMLUsageTracker(
                user_id="u-1",
                operation_type="t",
                effective_model_name="real-model-x",
            )
            log = _fake_function_log(provider=provider_str, client_name="Alias")
            provider, model_name = tracker._extract_model_info(log)
            assert provider == expected_enum, (
                f"{provider_str!r} mapped to {provider}, expected " f"{expected_enum}"
            )
            assert model_name == "real-model-x"

    def test_last_resort_fallback_uses_effective_name_on_probe_error(
        self,
    ) -> None:
        """Defensive: if probing function_log.calls raises (BAML
        SDK shape drift, partial fixture), the tracker still
        emits the routed model name when one was supplied. Better
        to record (BAML, real-model) than "unknown" — losing the
        model name eliminates cost-by-model analysis entirely.

        Using a real class with a throwing property (rather than
        mutating MagicMock's class) keeps test isolation clean —
        ``type(MagicMock()).calls = property(...)`` would leak the
        attribute to every MagicMock in the suite."""

        class _BrokenLog:
            @property
            def calls(self):
                raise AttributeError("malformed FunctionLog")

        tracker = BAMLUsageTracker(
            user_id="u-1",
            operation_type="t",
            effective_model_name="qwen2.5:14b",
        )
        provider, model_name = tracker._extract_model_info(_BrokenLog())
        assert provider == ModelProvider.BAML
        assert model_name == "qwen2.5:14b"

    def test_last_resort_fallback_unknown_without_override(self) -> None:
        """Mirror pin: when probing fails AND no override is set,
        fall through to the legacy 'unknown' name. That preserves
        the pre-fix behavior for callers that don't opt into the
        new plumbing."""

        class _BrokenLog:
            @property
            def calls(self):
                raise AttributeError("malformed FunctionLog")

        tracker = BAMLUsageTracker(user_id="u-1", operation_type="t")
        provider, model_name = tracker._extract_model_info(_BrokenLog())
        assert provider == ModelProvider.BAML
        assert model_name == "unknown"


class TestLLMUsageRequestModelName:
    """End-to-end pin: when extract_nodes_from_chunk runs with a
    routed model, the LLMUsageRequest sent to usage_tracking_service
    carries that model name (not the synthetic alias).

    Repro shape: stub the BAML execution + the usage_tracking_service
    so we can inspect the emitted LLMUsageRequest. Without this
    end-to-end pin, future refactors that drop the effective_model_name
    threading from one of the call sites would slip past the unit
    test on _extract_model_info."""

    @pytest.mark.asyncio
    async def test_emitted_request_carries_routed_model_name(self) -> None:
        from graphora_server.utils.baml_usage_tracker import (
            track_baml_extract_nodes_from_chunk,
        )

        fake_b = MagicMock()
        # b.ExtractNodesFromChunk returns an object with a .data
        # attribute that response_model.model_validate accepts.
        fake_result = MagicMock()
        fake_result.data = {}
        fake_b.ExtractNodesFromChunk = MagicMock(return_value=fake_result)

        fake_response_model = MagicMock()
        fake_response_model.model_validate = MagicMock(return_value="parsed")

        captured: dict[str, Any] = {}

        async def capture_track(*args, **kwargs):
            # ``track_llm_usage`` is called with keyword args
            # (``request=...``) by the tracker — accept both shapes
            # so this capture is robust to a future signature
            # tweak.
            captured["request"] = kwargs.get("request") or (args[0] if args else None)
            captured["kwargs"] = kwargs

        # Patch the BAML execution surface, the TypeBuilder (Rust
        # binding rejects MagicMocks at add_property), the
        # response_model parse helper, the Collector class, and the
        # usage_tracking_service hook. Each is invoked exactly once
        # inside the tracker — a future refactor that moves any of
        # them surfaces here.
        fake_log = _fake_function_log(
            provider="google-ai",
            client_name="DynamicGemini",  # synthetic alias
            input_tokens=120,
            output_tokens=80,
        )
        with (
            patch(
                "graphora_server.utils.baml_usage_tracker.usage_tracking_service"
            ) as svc,
            patch("graphora_server.baml_client.b", fake_b),
            patch(
                "graphora_server.baml_client.type_builder.TypeBuilder"
            ) as TB,
            patch(
                "graphora_server.utils.parse_pydantic_schema.build_from_pydantic"
            ) as build_pyd,
            patch(
                "graphora_server.utils.baml_usage_tracker.Collector"
            ) as col_class,
        ):
            svc.track_llm_usage = AsyncMock(side_effect=capture_track)
            # TypeBuilder() returns a MagicMock; tb.DynamicContainer
            # .add_property("data", res) becomes a no-op MagicMock
            # call instead of routing into the Rust FieldType check.
            TB.return_value = MagicMock()
            build_pyd.return_value = MagicMock()
            fake_collector = MagicMock()
            fake_collector.last = fake_log
            col_class.return_value = fake_collector

            await track_baml_extract_nodes_from_chunk(
                user_id="u-1",
                chunk="hello",
                response_model=fake_response_model,
                transform_id="tx-1",
                effective_model_name="gemini-2.5-pro",
            )

        # The emitted LLMUsageRequest must carry the routed model
        # name — NOT the synthetic alias from the FunctionLog.
        req = captured.get("request")
        assert req is not None, (
            "usage_tracking_service.track_llm_usage was never called — "
            "the tracker integration is broken. Check the patches."
        )
        assert req.model_name == "gemini-2.5-pro", (
            "LLMUsageRequest recorded the wrong model name. Pre-fix "
            "it would have been 'DynamicGemini' (the synthetic BAML "
            "alias), which collapses every routed model into one "
            f"bucket. Got: {req.model_name!r}"
        )
        assert req.input_tokens == 120
        assert req.output_tokens == 80
        assert req.transform_id == "tx-1"

    @pytest.mark.asyncio
    async def test_no_override_keeps_back_compat_alias(self) -> None:
        """Legacy callers (no effective_model_name) keep recording
        the synthetic alias — back-compat pin. The fix is additive,
        not behavior-changing for paths that don't opt in."""
        from graphora_server.utils.baml_usage_tracker import (
            track_baml_extract_nodes_from_chunk,
        )

        fake_b = MagicMock()
        fake_result = MagicMock()
        fake_result.data = {}
        fake_b.ExtractNodesFromChunk = MagicMock(return_value=fake_result)

        fake_response_model = MagicMock()
        fake_response_model.model_validate = MagicMock(return_value="parsed")

        captured: dict[str, Any] = {}

        async def capture_track(*args, **kwargs):
            # ``track_llm_usage`` is called with keyword args
            # (``request=...``) by the tracker — accept both shapes
            # so this capture is robust to a future signature
            # tweak.
            captured["request"] = kwargs.get("request") or (args[0] if args else None)
            captured["kwargs"] = kwargs

        fake_log = _fake_function_log(
            provider="google-ai", client_name="DynamicGemini"
        )
        with (
            patch(
                "graphora_server.utils.baml_usage_tracker.usage_tracking_service"
            ) as svc,
            patch("graphora_server.baml_client.b", fake_b),
            patch(
                "graphora_server.baml_client.type_builder.TypeBuilder"
            ) as TB,
            patch(
                "graphora_server.utils.parse_pydantic_schema.build_from_pydantic"
            ) as build_pyd,
            patch("graphora_server.utils.baml_usage_tracker.Collector") as col_class,
        ):
            svc.track_llm_usage = AsyncMock(side_effect=capture_track)
            TB.return_value = MagicMock()
            build_pyd.return_value = MagicMock()
            fake_collector = MagicMock()
            fake_collector.last = fake_log
            col_class.return_value = fake_collector

            await track_baml_extract_nodes_from_chunk(
                user_id="u-1",
                chunk="hello",
                response_model=fake_response_model,
                transform_id="tx-1",
                # No effective_model_name — back-compat path.
            )

        req = captured.get("request")
        assert req is not None
        assert req.model_name == "DynamicGemini"
