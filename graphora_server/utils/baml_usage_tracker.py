"""
BAML usage tracking utilities using BoundaryML Collector.

This module provides utilities to track token usage and costs
for BAML function calls using the official Collector API.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from baml_py import Collector
from graphora_server.services.usage_tracking import usage_tracking_service
from graphora_server.schemas.usage import LLMUsageRequest, ModelProvider
from graphora_server.utils.logger import logger


class BAMLUsageTracker:
    """
    Wrapper around BAML Collector to integrate with our usage tracking system
    """

    def __init__(
        self,
        user_id: str,
        operation_type: str,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
        operation_context: Optional[str] = None,
        collector_name: Optional[str] = None,
        effective_model_name: Optional[str] = None,
        effective_provider: Optional[ModelProvider] = None,
    ):
        self.user_id = user_id
        self.operation_type = operation_type
        self.transform_id = transform_id
        self.document_usage_id = document_usage_id
        self.operation_context = operation_context
        # Reviewer-flagged P2 on commit 71923d4 (B5-obs slice 3): the
        # FunctionLog's ``call.client_name`` is the synthetic BAML
        # alias (``DynamicGemini`` / ``DynamicOllama``), set at
        # registry construction time in
        # graphora_server/utils/llm_helper.py::create_baml_client_registry.
        # That means a primary ``gemini-1.5-flash`` call and a
        # refinement ``gemini-2.5-pro`` call would BOTH land in
        # llm_usage as the same synthetic name — useless for the
        # B5-obs cost reports and ``models_used`` aggregation.
        #
        # ``effective_model_name`` carries the actual routed model
        # from the LLM client (which got it from
        # get_baml_registry_for_user's returned tuple — that value
        # already reflects ``model_override`` when one was applied).
        # _extract_model_info prefers this when set; falls back to
        # the FunctionLog probe so callers that don't supply it
        # behave as before.
        self.effective_model_name = effective_model_name
        # Reviewer-flagged P2 on commit 89aee97: Ollama-backed BAML
        # calls go through BAML's ``openai-generic`` provider string
        # (because Ollama exposes an OpenAI-compatible API). The
        # FunctionLog-based provider inference therefore reports
        # ModelProvider.OPENAI for what is really an Ollama call —
        # so /cost shows ``openai:<ollama-model>`` rows. Threading
        # an explicit ``effective_provider`` from the LLM client
        # (which already knows the real provider from
        # get_baml_registry_for_user's tuple) lets _extract_model_info
        # report the truth. When unset, FunctionLog inference is
        # preserved for back-compat.
        self.effective_provider = effective_provider

        # Create BAML collector
        self.collector = Collector(name=collector_name or f"{operation_type}_{user_id}")

        # Timing
        self.start_time = datetime.now(timezone.utc)

    async def track_function_call(self, function_name: str, result: Any) -> None:
        """
        Track a BAML function call after it completes

        Args:
            function_name: Name of the BAML function called
            result: Result from the BAML function call
        """
        try:
            end_time = datetime.now(timezone.utc)

            if self.collector.last is None:
                logger.warning(f"No BAML usage data available for {function_name}")
                return

            # Get the last function log
            function_log = self.collector.last

            # Extract usage information
            usage = function_log.usage
            input_tokens = usage.input_tokens or 0
            output_tokens = usage.output_tokens or 0

            # Calculate latency
            latency_ms = None
            if function_log.timing and function_log.timing.duration_ms:
                latency_ms = function_log.timing.duration_ms
            else:
                latency_ms = int((end_time - self.start_time).total_seconds() * 1000)

            # Determine model information from the calls
            model_provider, model_name = self._extract_model_info(function_log)

            # Check if the call was successful
            success = len([call for call in function_log.calls if call.selected]) > 0
            error_message = None

            if not success and function_log.calls:
                # Try to extract error from failed calls
                failed_calls = [
                    call for call in function_log.calls if not call.selected
                ]
                if failed_calls and failed_calls[-1].http_response:
                    error_message = f"HTTP {failed_calls[-1].http_response.status}"

            # Create usage request. Pre-fix the model_provider here
            # was hardcoded ModelProvider.BAML — _extract_model_info
            # returned the correct provider but it was discarded.
            # That was harmless before (everything ended up as BAML
            # anyway) but with effective_provider plumbed in for
            # Ollama vs Gemini disambiguation, the model_provider on
            # the LLMUsageRequest needs to actually flow through.
            usage_request = LLMUsageRequest(
                transform_id=self.transform_id,
                document_usage_id=self.document_usage_id,
                model_provider=model_provider,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation_type=f"baml_{function_name}",
                latency_ms=latency_ms,
            )

            # Track the usage
            await usage_tracking_service.track_llm_usage(
                user_id=self.user_id,
                request=usage_request,
                request_timestamp=self.start_time,
                response_timestamp=end_time,
                success=success,
                error_message=error_message,
            )

            logger.info(
                f"Tracked BAML usage: {function_name} - {input_tokens + output_tokens} tokens"
            )

        except Exception as e:
            logger.error(f"Failed to track BAML usage for {function_name}: {str(e)}")

    def _extract_model_info(self, function_log) -> tuple[ModelProvider, str]:
        """
        Extract model provider and name from function log

        Args:
            function_log: BAML FunctionLog object

        Returns:
            Tuple of (ModelProvider, model_name)
        """
        try:
            if function_log.calls:
                # Get the selected call or the last call
                selected_calls = [call for call in function_log.calls if call.selected]
                call = selected_calls[0] if selected_calls else function_log.calls[-1]

                provider = call.provider.lower()
                # Prefer the routed model name when the caller
                # supplied one (B5-obs slice 3): ``call.client_name``
                # is the synthetic BAML alias which collapses every
                # call from the same registry into the same string,
                # making model-level cost analysis impossible. The
                # effective name is the real provider model
                # ("gemini-2.5-pro", "qwen2.5:14b", etc.) and is
                # what /cost should aggregate by.
                model_name = self.effective_model_name or call.client_name

                # Reviewer-flagged P2 on commit 89aee97: prefer the
                # caller-supplied ``effective_provider`` over BAML's
                # ``call.provider`` string. The Ollama path routes
                # through BAML's ``openai-generic`` provider (Ollama
                # exposes an OpenAI-compatible API), which the
                # infer-from-string branches below would map to
                # ModelProvider.OPENAI — wrong for cost reporting.
                # The LLM client knows the real provider via
                # get_baml_registry_for_user's tuple; threading it
                # through cuts out the lossy inference step.
                if self.effective_provider is not None:
                    return self.effective_provider, model_name

                # Map BAML providers to our ModelProvider enum
                if "openai" in provider:
                    return ModelProvider.OPENAI, model_name
                elif "anthropic" in provider or "claude" in provider:
                    return ModelProvider.ANTHROPIC, model_name
                elif (
                    "google" in provider or "gemini" in provider or "vertex" in provider
                ):
                    return ModelProvider.GEMINI, model_name
                else:
                    return ModelProvider.BAML, model_name

        except Exception as e:
            logger.warning(f"Could not extract model info from BAML call: {str(e)}")

        # Last-resort fallback: still surface the effective model
        # if the caller provided one, even when FunctionLog probing
        # failed. Better to record (effective_provider, model_name)
        # than lose either entirely. Prefer effective_provider when
        # set; fall through to BAML otherwise.
        if self.effective_model_name:
            return (
                self.effective_provider or ModelProvider.BAML,
                self.effective_model_name,
            )
        return self.effective_provider or ModelProvider.BAML, "unknown"

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get a summary of usage from the collector

        Returns:
            Dict containing usage summary
        """
        try:
            total_usage = self.collector.usage

            summary = {
                "total_input_tokens": total_usage.input_tokens or 0,
                "total_output_tokens": total_usage.output_tokens or 0,
                "total_tokens": (total_usage.input_tokens or 0)
                + (total_usage.output_tokens or 0),
                "total_calls": len(self.collector.logs),
                "functions_called": [log.function_name for log in self.collector.logs],
                "providers_used": (
                    list(
                        set(
                            [
                                call.provider
                                for log in self.collector.logs
                                for call in log.calls
                            ]
                        )
                    )
                    if self.collector.logs
                    else []
                ),
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to get BAML usage summary: {str(e)}")
            return {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_calls": 0,
                "functions_called": [],
                "providers_used": [],
            }


async def track_baml_function(
    user_id: str,
    function_name: str,
    operation_type: str,
    baml_function_call,
    *args,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    operation_context: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Execute a BAML function call with automatic usage tracking

    Args:
        user_id: User's ID
        function_name: Name of the BAML function
        operation_type: Type of operation being performed
        baml_function_call: The BAML function to call
        *args: Arguments to pass to the BAML function
        transform_id: Optional transformation ID
        document_usage_id: Optional document usage ID
        operation_context: Optional operation context
        **kwargs: Keyword arguments to pass to the BAML function

    Returns:
        Result from the BAML function call
    """
    # Create tracker
    tracker = BAMLUsageTracker(
        user_id=user_id,
        operation_type=operation_type,
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        operation_context=operation_context,
        collector_name=f"{function_name}_{user_id}",
    )

    try:
        # Add collector to BAML options
        baml_options = kwargs.get("baml_options", {})
        baml_options["collector"] = tracker.collector
        kwargs["baml_options"] = baml_options

        # Execute the BAML function
        result = baml_function_call(*args, **kwargs)

        # Track the usage
        await tracker.track_function_call(function_name, result)

        return result

    except Exception:
        # Track failed call
        await tracker.track_function_call(function_name, None)
        raise


async def track_baml_extract_nodes_from_chunk(
    user_id: str,
    chunk: str,
    response_model,
    ontology_yaml: Optional[str] = None,
    context: str = "",
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    client_registry=None,
    effective_model_name: Optional[str] = None,
    effective_provider: Optional[ModelProvider] = None,
):
    """
    Track BAML ExtractNodesFromChunk function call.

    ``effective_model_name`` (B5-obs slice 3): the model actually
    routed for this call. When the multi-pass extractor's refinement
    pass overrides the model, this carries the override so the
    emitted LLMUsageRequest records the real model name, not the
    synthetic BAML registry alias.

    ``effective_provider`` (P2 fix on commit 89aee97): the provider
    enum value matching the routed model. Bypasses BAML's
    ``call.provider`` string for cost-reporting purposes — needed
    because Ollama's openai-generic provider would otherwise be
    misreported as ``ModelProvider.OPENAI``.
    """
    from graphora_server.baml_client import b
    from graphora_server.baml_client.type_builder import TypeBuilder
    from graphora_server.utils.parse_pydantic_schema import build_from_pydantic

    # Create tracker
    tracker = BAMLUsageTracker(
        user_id=user_id,
        operation_type="chunk_entity_extraction",
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        operation_context=f"chunk_processing:{len(chunk)} chars",
        collector_name=f"extract_nodes_{user_id}",
        effective_model_name=effective_model_name,
        effective_provider=effective_provider,
    )

    try:
        # Prepare TypeBuilder
        tb = TypeBuilder()
        res = build_from_pydantic(response_model, tb)
        tb.DynamicContainer.add_property("data", res)

        # Execute with collector and client registry
        baml_options = {"tb": tb, "collector": tracker.collector}
        if client_registry:
            baml_options["client_registry"] = client_registry

        result = b.ExtractNodesFromChunk(chunk, context, baml_options=baml_options)

        # Track usage
        await tracker.track_function_call("ExtractNodesFromChunk", result)

        return response_model.model_validate(result.data)

    except Exception:
        await tracker.track_function_call("ExtractNodesFromChunk", None)
        raise


async def track_baml_extract_relationships_from_chunk(
    user_id: str,
    chunk: str,
    response_model,
    ontology_yaml: Optional[str] = None,
    context: str = "",
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    client_registry=None,
    effective_model_name: Optional[str] = None,
    effective_provider: Optional[ModelProvider] = None,
):
    """
    Track BAML ExtractRelationshipsFromChunk function call.

    See ``track_baml_extract_nodes_from_chunk`` for the
    ``effective_model_name`` and ``effective_provider`` contracts.
    """
    from graphora_server.baml_client import b
    from graphora_server.baml_client.type_builder import TypeBuilder
    from graphora_server.utils.parse_pydantic_schema import build_from_pydantic

    # Create tracker
    tracker = BAMLUsageTracker(
        user_id=user_id,
        operation_type="chunk_relationship_extraction",
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        operation_context=f"chunk_processing:{len(chunk)} chars",
        collector_name=f"extract_relationships_{user_id}",
        effective_model_name=effective_model_name,
        effective_provider=effective_provider,
    )

    try:
        # Prepare TypeBuilder
        tb = TypeBuilder()
        res = build_from_pydantic(response_model, tb)
        tb.DynamicContainer.add_property("data", res)

        # Execute with collector and client registry
        baml_options = {"tb": tb, "collector": tracker.collector}
        if client_registry:
            baml_options["client_registry"] = client_registry

        result = b.ExtractRelationshipsFromChunk(
            chunk, context, baml_options=baml_options
        )

        # Track usage
        await tracker.track_function_call("ExtractRelationshipsFromChunk", result)

        return response_model.model_validate(result.data)

    except Exception:
        await tracker.track_function_call("ExtractRelationshipsFromChunk", None)
        raise


async def track_baml_infer_relationship(
    user_id: str,
    rel_type: str,
    source_type: str = "",
    source_entities: str = "",
    target_type: str = "",
    target_entities: str = "",
    existing_rels: str = "",
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    client_registry=None,
):
    """
    Track BAML InferRelationship function call
    """
    from graphora_server.baml_client import b

    # Create tracker
    tracker = BAMLUsageTracker(
        user_id=user_id,
        operation_type="relationship_inference",
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        operation_context=f"infer:{rel_type}",
        collector_name=f"infer_relationship_{user_id}",
    )

    try:
        # Execute with collector and client registry
        baml_options = {"collector": tracker.collector}
        if client_registry:
            baml_options["client_registry"] = client_registry

        result = b.InferRelationship(
            rel_type=rel_type,
            source_type=source_type,
            source_entities=source_entities,
            target_type=target_type,
            target_entities=target_entities,
            existing_rels=existing_rels,
            baml_options=baml_options,
        )

        # Track usage
        await tracker.track_function_call("InferRelationship", result)

        return result

    except Exception:
        await tracker.track_function_call("InferRelationship", None)
        raise


async def track_baml_standardise_properties(
    user_id: str,
    entity_group_type: str,
    entities_json: str,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    client_registry=None,
):
    """
    Track BAML StandardiseProperties function call
    """
    from graphora_server.baml_client import b

    # Create tracker
    tracker = BAMLUsageTracker(
        user_id=user_id,
        operation_type="property_standardization",
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        operation_context=f"standardize:{entity_group_type}",
        collector_name=f"standardise_properties_{user_id}",
    )

    try:
        # Execute with collector and client registry
        baml_options = {"collector": tracker.collector}
        if client_registry:
            baml_options["client_registry"] = client_registry

        result = b.StandardiseProperties(
            entity_group_type=entity_group_type,
            entities_json=entities_json,
            baml_options=baml_options,
        )

        # Track usage
        await tracker.track_function_call("StandardiseProperties", result)

        return result

    except Exception:
        await tracker.track_function_call("StandardiseProperties", None)
        raise


async def track_baml_resolve_entities(
    user_id: str,
    entity_type: str,
    node_dicts_str: str,
    transform_id: Optional[str] = None,
    document_usage_id: Optional[str] = None,
    client_registry=None,
):
    """
    Track BAML ResolveEntities function call
    """
    from graphora_server.baml_client import b

    # Create tracker
    tracker = BAMLUsageTracker(
        user_id=user_id,
        operation_type="entity_resolution",
        transform_id=transform_id,
        document_usage_id=document_usage_id,
        operation_context=f"resolve:{entity_type}",
        collector_name=f"resolve_entities_{user_id}",
    )

    try:
        # Execute with collector and client registry
        baml_options = {"collector": tracker.collector}
        if client_registry:
            baml_options["client_registry"] = client_registry

        result = b.ResolveEntities(
            entity_type=entity_type,
            node_dicts_str=node_dicts_str,
            baml_options=baml_options,
        )

        # Track usage
        await tracker.track_function_call("ResolveEntities", result)

        return result

    except Exception:
        await tracker.track_function_call("ResolveEntities", None)
        raise


async def track_baml_get_matching_nodes(
    user_id: str,
    candidate_sets: List[str],
    merge_id: Optional[str] = None,
    transform_id: Optional[str] = None,
    client_registry=None,
):
    """
    Track BAML GetMatchingNodes function call for merge operations
    """
    from graphora_server.baml_client import b

    # Create tracker
    tracker = BAMLUsageTracker(
        user_id=user_id,
        operation_type="merge_entity_matching",
        transform_id=transform_id,
        operation_context=f"merge:{merge_id}",
        collector_name=f"get_matching_nodes_{user_id}",
    )

    try:
        # Execute with collector and client registry
        baml_options = {"collector": tracker.collector}
        if client_registry:
            baml_options["client_registry"] = client_registry

        result = b.GetMatchingNodes(
            candidate_sets=candidate_sets, baml_options=baml_options
        )

        # Track usage
        await tracker.track_function_call("GetMatchingNodes", result)

        return result

    except Exception:
        await tracker.track_function_call("GetMatchingNodes", None)
        raise


async def track_baml_eval_changes(
    user_id: str,
    change_logs: str,
    past_resolutions: str,
    merge_id: Optional[str] = None,
    transform_id: Optional[str] = None,
    ontology_id: Optional[str] = None,
    client_registry=None,
):
    """
    Track BAML EvalChanges function call for merge conflict analysis
    """
    from graphora_server.baml_client import b

    # Create tracker
    tracker = BAMLUsageTracker(
        user_id=user_id,
        operation_type="merge_conflict_analysis",
        transform_id=transform_id,
        operation_context=f"merge:{merge_id}:ontology:{ontology_id}",
        collector_name=f"eval_changes_{user_id}",
    )

    try:
        # Execute with collector and client registry
        baml_options = {"collector": tracker.collector}
        if client_registry:
            baml_options["client_registry"] = client_registry

        result = b.EvalChanges(
            change_logs=change_logs,
            past_resolutions=past_resolutions,
            baml_options=baml_options,
        )

        # Track usage
        await tracker.track_function_call("EvalChanges", result)

        return result

    except Exception:
        await tracker.track_function_call("EvalChanges", None)
        raise
