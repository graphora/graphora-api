from typing import List, Type, Optional, Any
from datetime import datetime, timezone
from pathlib import Path
from collections import OrderedDict
import asyncio
from copy import deepcopy
import json

from graphora_server.baml_client.type_builder import TypeBuilder
from pydantic import BaseModel
from graphora_server.baml_client import b
from graphora_server.baml_client.types import (
    RelationshipInference,
    ResolvedEntities,
    StandardisedProperties,
)
from graphora_server.utils.parse_pydantic_schema import build_from_pydantic
from graphora_server.baml_client import reset_baml_env_vars

# get_prompt_version is imported lazily at each callsite below —
# importing at module top would route through
# graphora_server.services.extraction/__init__.py which transitively
# re-imports transform.helpers (via multi_pass_extractor) during
# its own import. Since helpers.py imports THIS module at its top,
# the chain forms a cycle on the helpers→client→extraction→helpers
# path. Lazy import dodges that — it's a tiny per-call cost (one
# attribute lookup) for clean module-load semantics.
import os
import dotenv

import hashlib
import pathlib
from google.genai import types
from graphora_server.utils.func_helper import retry_async
from graphora_server.utils.llm_usage_tracker import track_gemini_usage
from graphora_server.utils.logger import logger
from graphora_server.utils.llm_helper import (
    create_gemini_client,
    get_baml_registry_for_user,
    get_user_llm_credentials,
)
from graphora_server.config import settings

dotenv.load_dotenv()
reset_baml_env_vars(dict(os.environ))


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


class _AsyncLRUCache:
    """Simple asyncio-safe LRU cache for storing serialized LLM responses."""

    def __init__(self, max_size: int = 128) -> None:
        self._max_size = max_size
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            value = self._data.get(key)
            if value is None:
                return None
            self._data.move_to_end(key)
            return deepcopy(value)

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._data[key] = deepcopy(value)
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)


class _RedisCache:
    """Async Redis-backed cache with JSON serialisation."""

    def __init__(self, url: str, namespace: str, ttl_seconds: Optional[int]):
        try:  # pragma: no cover - optional dependency
            from redis.asyncio import Redis  # type: ignore
        except Exception as exc:  # pragma: no cover - handled via fallback
            raise RuntimeError("redis.asyncio is required for Redis LLM cache") from exc

        self._client = Redis.from_url(url, encoding="utf-8", decode_responses=True)
        self._namespace = namespace
        self._ttl = ttl_seconds

    def _namespaced(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        raw = await self._client.get(self._namespaced(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:  # pragma: no cover - defensive cleanup
            await self._client.delete(self._namespaced(key))
            logger.warning(
                "Failed to decode cached payload; purging entry",
                extra={"cache_key": key, "namespace": self._namespace},
            )
            return None

    async def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        await self._client.set(
            self._namespaced(key),
            payload,
            ex=self._ttl if self._ttl and self._ttl > 0 else None,
        )


def _create_cache(namespace: str):
    max_entries = getattr(settings, "LLM_CACHE_MAX_ENTRIES", 128)
    cache_url = getattr(settings, "LLM_CACHE_URL", None)
    if cache_url:
        ttl_seconds = max(getattr(settings, "CACHE_TTL_HOURS", 24), 0) * 3600
        try:
            logger.debug(
                "Initialising Redis-backed LLM cache",
                extra={"namespace": namespace, "cache_url": cache_url},
            )
            return _RedisCache(cache_url, namespace, ttl_seconds)
        except Exception as exc:  # pragma: no cover - fallback when Redis unavailable
            logger.warning(
                "Redis cache unavailable for namespace %s: %s. Falling back to in-memory cache.",
                namespace,
                exc,
            )
    return _AsyncLRUCache(max_size=max_entries)


_PDF_NODE_CACHE = _create_cache("pdf-nodes")
_PDF_REL_CACHE = _create_cache("pdf-relationships")
_CHUNK_NODE_CACHE = _create_cache("chunk-nodes")
_CHUNK_REL_CACHE = _create_cache("chunk-relationships")


def _cache_key(*parts: str) -> str:
    normalized = "||".join(part or "" for part in parts)
    return md5(normalized)


def _get_prompt_version(baml_function_name: str) -> str:
    """Lazy proxy for prompt_versions.get_prompt_version.

    Module-level import of prompt_versions would route through
    services/extraction/__init__.py which transitively re-imports
    transform/helpers.py via multi_pass_extractor — and since
    helpers.py imports THIS module at its top, that closes a cycle
    on the helpers→client→extraction→helpers path. Lazy import
    dodges the cycle at one attribute-lookup of cost per call.

    Tests that need to override the version (e.g., the cache-key
    versioning suite) should monkeypatch this shim directly via
    ``monkeypatch.setattr(llm_client, "_get_prompt_version", ...)``."""
    from graphora_server.services.extraction.prompt_versions import (
        get_prompt_version,
    )

    return get_prompt_version(baml_function_name) or ""


def _preview(text: Optional[str], limit: int = 200) -> str:
    if not text:
        return ""
    clean = text.replace("\n", " ")
    return clean[:limit] + ("…" if len(clean) > limit else "")


def _provider_string_to_enum(provider: Optional[str]):
    """Map the provider string returned by
    ``get_baml_registry_for_user`` to the ``ModelProvider`` enum
    used in usage tracking.

    Reviewer-flagged P2 on commit 89aee97: BAML's
    ``call.provider`` string is the *transport* provider (e.g.,
    ``openai-generic`` for Ollama because Ollama exposes an
    OpenAI-compatible API). The string returned by
    get_baml_registry_for_user is the *logical* provider — what
    the user actually configured. Threading the logical mapping
    here keeps the cost-reporting layer honest about which
    provider was used, not which transport.

    Returns ``None`` for unknown/missing provider strings so the
    tracker falls through to its existing FunctionLog-based
    inference (preserving back-compat for the legacy paths that
    don't supply a provider).
    """
    # Lazy import to avoid pulling the schemas module at LLM client
    # module load (the schemas package imports pydantic chains that
    # are heavy at import time).
    from graphora_server.schemas.usage import ModelProvider

    if not provider:
        return None
    normalized = provider.lower()
    if normalized == "gemini":
        return ModelProvider.GEMINI
    if normalized == "ollama":
        return ModelProvider.OLLAMA
    if normalized == "openai":
        return ModelProvider.OPENAI
    if normalized == "anthropic":
        return ModelProvider.ANTHROPIC
    return None


class LLMClient:
    """Client for LLM-based extraction"""

    # @cached(ttl=86400,
    #     key_builder=lambda f, *args, **kwargs: f"{md5(args[1])+':'+str(kwargs['response_model'])}")
    @retry_async(
        max_attempts=5, delay=2, backoff=2, exceptions=(ValueError, Exception)
    )  # Adjust exceptions
    async def extract_nodes_from_pdf(
        self,
        pdf_path: str,
        response_model: Type[BaseModel],
        ontology_yaml: str,
        context: str = "",
        model_id: str = None,
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> BaseModel:
        """Extract entities and relationships from PDF"""
        if not user_id:
            raise ValueError("user_id is required to get LLM credentials")

        # Get user's LLM credentials
        api_key, model_name = await get_user_llm_credentials(user_id)

        # Use provided model_id or default to user's configured model
        model_to_use = model_id if model_id else model_name

        # Create Gemini client with user's API key
        client = create_gemini_client(api_key)

        filepath = pathlib.Path(pdf_path)
        file_bytes = filepath.read_bytes()
        file = types.Part.from_bytes(
            data=file_bytes,
            mime_type="application/pdf",
        )
        content_hash = hashlib.md5(file_bytes).hexdigest()
        ontology_hash = md5(ontology_yaml or "")
        context_hash = md5(context or "")
        # Prompt version in the cache key so an output-shape change
        # (Gate 4 v1.0.0 → v1.1.0 added source_excerpt) auto-
        # invalidates pre-existing cached responses. Without this a
        # v1.0.0 cache entry would validate against the v1.1.0
        # schema (source_excerpt is Optional and defaults to None)
        # and silently serve evidence-less responses to v1.1.0
        # callers. Reviewer caught this on the Gate-4 commit.
        cache_key = _cache_key(
            "pdf-nodes",
            user_id or "",
            model_to_use,
            content_hash,
            ontology_hash,
            context_hash,
            _get_prompt_version("ExtractNodesFromPdf"),
        )

        cached = await _PDF_NODE_CACHE.get(cache_key)
        if cached:
            logger.debug(
                "Using cached Gemini PDF entity extraction",
                extra={
                    "transform_id": transform_id,
                    "file": filepath.name,
                    "cache_key": cache_key,
                },
            )
            return response_model.model_validate(cached)

        # Generate a structured response using the Gemini API
        prompt = f"""
        Extract structured information from the PDF file according to the ontology specification.
        <ontology>
        {ontology_yaml}
        </ontology>

        Format the output as a JSON object with the following structure:
        1. For each entity type, include a list field named "<entity_type>_list" containing all instances
        2. Include metadata fields:
            - extraction_timestamp: Current timestamp in ISOformat string
            - tokens_used: Number of tokens used (if available)
            - confidence_score: Overall confidence in extraction (0.0 to 1.0)
        3. Omit optional fields if information is not clearly present
        4. No additional properties. Just the specified fields.

        When extracting new information, maintain consistency with these previously identified entities. If previous Entities miss any properties add them.
        These Nodes were identified from the previous text chunks of the same doc. Use the `id` field to refer & match the nodes below.
        ```
        {context}
        ```

        <rules>
        - For name fields, sometimes names may be written as `LASTNAME, FIRSTNAME`. So interpret accordingly
        - Only extract information that is explicitly present in the text
        - Set confidence scores based on certainty of extraction
        - Include all required fields for each node
        - Omit optional fields if information is not clearly present
        </rules>

        <per_entity_source_excerpt>
        For EACH extracted entity, include a `source_excerpt` field —
        a 1-2 sentence VERBATIM quote from the PDF where this entity
        is mentioned. This powers the Evidence-tab provenance
        surface for end users.

        Rules:
        - Use the EXACT WORDS from the PDF text, not a paraphrase.
        - Choose the most informative single sentence (or two
          adjacent sentences) that mentions the entity.
        - Keep it short — ideally under 200 characters.
        - If the entity is mentioned on multiple pages, pick the
          mention with the most identifying information (full name,
          role, etc.).
        - If you cannot identify a specific source sentence, omit
          the field rather than fabricating one.
        </per_entity_source_excerpt>
        """
        request_timestamp = datetime.now(timezone.utc)
        logger.debug(
            "Gemini PDF extraction request",
            extra={
                "transform_id": transform_id,
                "file": filepath.name,
                "prompt_preview": _preview(prompt),
                "context_length": len(context or ""),
            },
        )
        response = client.models.generate_content(
            model=model_to_use,
            contents=[file, prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": response_model,
                "temperature": 0.0,
                "top_p": 0.0,
                "top_k": 1,
                "candidate_count": 1,
            },
        )
        response_timestamp = datetime.now(timezone.utc)

        # Track usage if user_id provided
        if user_id:
            try:
                await track_gemini_usage(
                    user_id=user_id,
                    model_name=model_to_use,
                    operation_type="pdf_entity_extraction",
                    response=response,
                    transform_id=transform_id,
                    document_usage_id=document_usage_id,
                    operation_context=f"pdf_processing:{Path(pdf_path).name}",
                    request_timestamp=request_timestamp,
                    response_timestamp=response_timestamp,
                )
            except Exception as e:
                logger.error(f"Failed to track Gemini usage: {str(e)}")

        # Convert the response to the pydantic model and return it
        logger.debug("*" * 30)
        logger.debug(response)
        logger.debug("*" * 30)
        logger.debug(response.usage_metadata)
        logger.debug("*" * 30)
        output_perc = 3
        if response.usage_metadata:
            output_perc = (
                response.usage_metadata.candidates_token_count * 100
            ) / response.usage_metadata.total_token_count

        # Handle empty responses gracefully. Do NOT cache the empty
        # result — Gemini's parser failures are transient (truncated
        # output, rate-limit JSON, etc.). Caching them would poison
        # the cache permanently for this (pdf_hash + ontology_hash +
        # context_hash) tuple, silently returning empty forever even
        # after the underlying transient issue resolves. Let the
        # next call retry the live API.
        if response.parsed is None:
            logger.warning(
                "Gemini PDF entity extraction returned no parseable response, "
                "returning empty model (NOT cached — will retry next call). "
                "This may indicate ontology-document mismatch or a transient "
                "Gemini parse failure.",
                extra={
                    "transform_id": transform_id,
                    "file": filepath.name,
                    "output_percentage": output_perc,
                },
            )
            return response_model()

        if output_perc < 4:
            logger.warning(
                "Gemini PDF entity response very small compared to prompt. "
                "This may indicate the ontology doesn't match the document content.",
                extra={
                    "transform_id": transform_id,
                    "file": filepath.name,
                    "output_percentage": output_perc,
                },
            )
        result_model = response.parsed
        await _PDF_NODE_CACHE.set(cache_key, result_model.model_dump(mode="json"))
        logger.debug(
            "Gemini PDF extraction response",
            extra={
                "transform_id": transform_id,
                "file": filepath.name,
                "response_preview": _preview(getattr(response, "text", "")),
                "total_tokens": getattr(
                    response.usage_metadata, "total_token_count", None
                ),
                "cache_key": cache_key,
            },
        )
        return result_model

    # @cached(ttl=86400,
    #     key_builder=lambda f, *args, **kwargs: f"{md5(args[1])+':'+str(kwargs['response_model'])}")
    @retry_async(
        max_attempts=5, delay=2, backoff=2, exceptions=(ValueError, Exception)
    )  # Adjust exceptions
    async def extract_relationships_from_pdf(
        self,
        pdf_path: str,
        response_model: Type[BaseModel],
        ontology_yaml: str,
        context: str = "",
        model_id: str = None,
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> BaseModel:
        """Extract entities and relationships from PDF"""
        if not user_id:
            raise ValueError("user_id is required to get LLM credentials")

        # Get user's LLM credentials
        api_key, model_name = await get_user_llm_credentials(user_id)

        # Use provided model_id or default to user's configured model
        model_to_use = model_id if model_id else model_name

        # Create Gemini client with user's API key
        client = create_gemini_client(api_key)

        filepath = pathlib.Path(pdf_path)
        file_bytes = filepath.read_bytes()
        file = types.Part.from_bytes(
            data=file_bytes,
            mime_type="application/pdf",
        )
        content_hash = hashlib.md5(file_bytes).hexdigest()
        ontology_hash = md5(ontology_yaml or "")
        context_hash = md5(context or "")
        # See pdf-nodes cache_key above for the prompt-version
        # rationale — same Gate-4 v1.0.0 → v1.1.0 invalidation.
        cache_key = _cache_key(
            "pdf-relationships",
            user_id or "",
            model_to_use,
            content_hash,
            ontology_hash,
            context_hash,
            _get_prompt_version("ExtractRelationshipsFromPdf"),
        )

        cached = await _PDF_REL_CACHE.get(cache_key)
        if cached:
            logger.debug(
                "Using cached Gemini PDF relationship extraction",
                extra={
                    "transform_id": transform_id,
                    "file": filepath.name,
                    "cache_key": cache_key,
                },
            )
            return response_model.model_validate(cached)
        # Generate a structured response using the Gemini API
        prompt = f"""
        Extract structured information from the PDF file according to the ontology specification.
        <ontology>
        {ontology_yaml}
        </ontology>

        These nodes & relationships were identified from the previous text chunks of the same doc.
        Format: (:source_node_type {{node_id & properties}}) - [:relationship_type {{properties}}] -> (:target_node_type {{node_id & properties}})
        ```
        {context}
        ```

        Output a JSON object following these rules:
        1. "<source>_<relationship>_<target>" for each relationship (e.g., "Company_HAS_BUSINESS_Business").
        2. Each relationship MUST include:
        - "source_id": The ID of the source node from the provided entities.
        - "target_id": The ID of the target node from the provided entities.
        - "properties": Any additional relationship properties (optional).
        - "source_excerpt": A 1-2 sentence VERBATIM quote from the
          PDF where this relationship is described (see rules below).
        3. Metadata: "extraction_timestamp" (ISO), "tokens_used", "confidence_score" (0.0-1.0).

        Remember:
        - Only extract information that is explicitly present in the text
        - Set confidence scores based on certainty of extraction
        - Include all required fields for each relationship
        - Omit optional fields if information is not clearly present

        <per_relationship_source_excerpt>
        For EACH extracted relationship, include a `source_excerpt`
        field — a 1-2 sentence VERBATIM quote from the PDF where the
        relationship is described. This powers the Evidence-tab
        provenance surface for end users.

        Rules:
        - Use the EXACT WORDS from the PDF text, not a paraphrase.
        - Choose the sentence that most clearly establishes the
          relationship between the source and target entities.
        - Keep it short — ideally under 200 characters.
        - If the relationship is implied across multiple sentences,
          pick the one that most directly connects the two entities.
        - If you cannot identify a specific source sentence, omit
          the field rather than fabricating one.
        </per_relationship_source_excerpt>
        """
        request_timestamp = datetime.now(timezone.utc)
        logger.debug(
            "Gemini PDF relationship request",
            extra={
                "transform_id": transform_id,
                "file": filepath.name,
                "prompt_preview": _preview(prompt),
                "context_length": len(context or ""),
            },
        )
        response = client.models.generate_content(
            model=model_to_use,
            contents=[file, prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": response_model,
                "temperature": 0.0,
                "top_p": 0.0,
                "top_k": 1,
                "candidate_count": 1,
            },
        )
        response_timestamp = datetime.now(timezone.utc)

        # Track usage if user_id provided
        if user_id:
            try:
                await track_gemini_usage(
                    user_id=user_id,
                    model_name=model_to_use,
                    operation_type="pdf_relationship_extraction",
                    response=response,
                    transform_id=transform_id,
                    document_usage_id=document_usage_id,
                    operation_context=f"pdf_processing:{Path(pdf_path).name}",
                    request_timestamp=request_timestamp,
                    response_timestamp=response_timestamp,
                )
            except Exception as e:
                logger.error(f"Failed to track Gemini usage: {str(e)}")

        # Convert the response to the pydantic model and return it
        logger.debug("*" * 30)
        logger.debug(response)
        logger.debug("*" * 30)
        logger.debug(response.usage_metadata)
        logger.debug("*" * 30)
        output_perc = 3
        if response.usage_metadata:
            output_perc = (
                response.usage_metadata.candidates_token_count * 100
            ) / response.usage_metadata.total_token_count

        # Empty/unparseable responses are NOT cached. Gemini sometimes
        # returns parsed=None on transient failures (output truncation,
        # rate-limit JSON, structured-output validator hiccups). If we
        # cache the empty model under this cache_key, every subsequent
        # call with the same (pdf_hash + ontology_hash + context_hash)
        # returns empty from cache forever — relationships never come
        # back even after the transient issue resolves.
        #
        # Concrete incident this guard remediates: a single transient
        # parse failure silently turned 7 relationships into 0 across
        # every retry of the same Apple 10K transform.
        if response.parsed is None:
            logger.warning(
                "Gemini PDF relationship extraction returned no parseable response, "
                "returning empty model (NOT cached — will retry next call). "
                "This may indicate ontology-document mismatch or a transient "
                "Gemini parse failure.",
                extra={
                    "transform_id": transform_id,
                    "file": filepath.name,
                    "output_percentage": output_perc,
                },
            )
            return response_model()

        if output_perc < 4:
            logger.warning(
                "Gemini PDF relationship response very small compared to prompt. "
                "This may indicate the ontology doesn't match the document content.",
                extra={
                    "transform_id": transform_id,
                    "file": filepath.name,
                    "output_percentage": output_perc,
                },
            )
        result_model = response.parsed
        await _PDF_REL_CACHE.set(cache_key, result_model.model_dump(mode="json"))
        logger.debug(
            "Gemini PDF relationship response",
            extra={
                "transform_id": transform_id,
                "file": filepath.name,
                "response_preview": _preview(getattr(response, "text", "")),
                "total_tokens": getattr(
                    response.usage_metadata, "total_token_count", None
                ),
                "cache_key": cache_key,
            },
        )
        return result_model

    # @cached(ttl=86400,
    #     key_builder=lambda f, *args, **kwargs: f"{md5(args[1])+':'+str(kwargs['response_model'])}")
    async def extract_nodes_from_chunk(
        self,
        chunk: str,
        response_model: Type[BaseModel],
        ontology_yaml: Optional[str] = None,
        context: str = "",
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> BaseModel:
        """Extract entities from text chunk.

        ``model_override`` swaps the model used for this call without
        changing the user's stored provider/auth — B5-obs slice 3
        uses this for refinement-pass model routing. None preserves
        the user's configured model.
        """
        if not user_id:
            raise ValueError("user_id is required to get LLM credentials")

        # Get user's LLM credentials and create client registry. The
        # effective model_name returned by get_baml_registry_for_user
        # reflects the override (when provided) — feed THAT into the
        # cache key so refinement-pass calls don't collide with
        # primary-pass calls in the chunk-nodes cache.
        client_registry, model_name, provider = await get_baml_registry_for_user(
            user_id, model_override=model_override
        )

        chunk_hash = md5(chunk)
        ontology_hash = md5(ontology_yaml or "")
        context_hash = md5(context or "")
        # See pdf-nodes cache_key above for the prompt-version
        # rationale — same Gate-4 v1.0.0 → v1.1.0 invalidation.
        cache_key = _cache_key(
            "chunk-nodes",
            user_id or "",
            model_name,
            chunk_hash,
            context_hash,
            ontology_hash,
            _get_prompt_version("ExtractNodesFromChunk"),
        )

        cached = await _CHUNK_NODE_CACHE.get(cache_key)
        if cached:
            logger.debug(
                "Using cached chunk entity extraction",
                extra={
                    "transform_id": transform_id,
                    "chunk_chars": len(chunk),
                    "cache_key": cache_key,
                },
            )
            return response_model.model_validate(cached)

        logger.debug(
            "BAML chunk entity extraction request",
            extra={
                "transform_id": transform_id,
                "chunk_preview": _preview(chunk),
                "context_length": len(context or ""),
                "ontology_hash": ontology_hash,
            },
        )

        if user_id:
            from graphora_server.utils.baml_usage_tracker import (
                track_baml_extract_nodes_from_chunk,
            )

            result_model = await track_baml_extract_nodes_from_chunk(
                user_id=user_id,
                chunk=chunk,
                response_model=response_model,
                ontology_yaml=ontology_yaml,
                context=context,
                transform_id=transform_id,
                document_usage_id=document_usage_id,
                client_registry=client_registry,
                # B5-obs slice 3: ``model_name`` here is the
                # EFFECTIVE model returned by
                # get_baml_registry_for_user (already reflects
                # ``model_override`` when one was applied). Threading
                # it to the tracker means llm_usage records the real
                # routed model name, not the synthetic BAML alias.
                effective_model_name=model_name,
                # P2 fix on commit 89aee97: also thread the effective
                # provider so Ollama (which BAML reports as
                # ``openai-generic``) doesn't get misclassified as
                # ModelProvider.OPENAI in cost reports.
                effective_provider=_provider_string_to_enum(provider),
            )
        else:
            tb = TypeBuilder()
            res = build_from_pydantic(response_model, tb)
            tb.DynamicContainer.add_property("data", res)
            result = b.ExtractNodesFromChunk(
                chunk,
                context,
                {"tb": tb, "client_registry": client_registry},
            )
            result_model = response_model.model_validate(result.data)

        await _CHUNK_NODE_CACHE.set(cache_key, result_model.model_dump(mode="json"))
        logger.debug(
            "BAML chunk entity extraction response",
            extra={
                "transform_id": transform_id,
                "chunk_chars": len(chunk),
                "cached": False,
                "cache_key": cache_key,
            },
        )
        return result_model

    # @cached(ttl=86400,
    #     key_builder=lambda f, *args, **kwargs: f"{md5(args[1])+':'+str(kwargs['response_model'])}")
    async def extract_relationships_from_chunk(
        self,
        chunk: str,
        response_model: Type[BaseModel],
        ontology_yaml: Optional[str] = None,
        context: str = "",
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> BaseModel:
        """Extract relationships from text chunk. See
        ``extract_nodes_from_chunk`` for the ``model_override``
        contract (B5-obs slice 3 refinement-pass routing)."""
        if not user_id:
            raise ValueError("user_id is required to get LLM credentials")

        # Get user's LLM credentials and create client registry.
        # model_override is forwarded to get_baml_registry_for_user;
        # returned model_name reflects the override and feeds the
        # cache key so refinement vs primary results don't collide.
        client_registry, model_name, provider = await get_baml_registry_for_user(
            user_id, model_override=model_override
        )

        chunk_hash = md5(chunk)
        ontology_hash = md5(ontology_yaml or "")
        context_hash = md5(context or "")
        # See pdf-nodes cache_key above for the prompt-version
        # rationale — same Gate-4 v1.0.0 → v1.1.0 invalidation.
        cache_key = _cache_key(
            "chunk-relationships",
            user_id or "",
            model_name,
            chunk_hash,
            context_hash,
            ontology_hash,
            _get_prompt_version("ExtractRelationshipsFromChunk"),
        )

        cached = await _CHUNK_REL_CACHE.get(cache_key)
        if cached:
            logger.debug(
                "Using cached chunk relationship extraction",
                extra={
                    "transform_id": transform_id,
                    "chunk_chars": len(chunk),
                    "cache_key": cache_key,
                },
            )
            return response_model.model_validate(cached)

        logger.debug(
            "BAML chunk relationship extraction request",
            extra={
                "transform_id": transform_id,
                "chunk_preview": _preview(chunk),
                "context_length": len(context or ""),
                "ontology_hash": ontology_hash,
            },
        )

        if user_id:
            from graphora_server.utils.baml_usage_tracker import (
                track_baml_extract_relationships_from_chunk,
            )

            result_model = await track_baml_extract_relationships_from_chunk(
                user_id=user_id,
                chunk=chunk,
                response_model=response_model,
                ontology_yaml=ontology_yaml,
                context=context,
                transform_id=transform_id,
                document_usage_id=document_usage_id,
                client_registry=client_registry,
                # B5-obs slice 3: mirror the nodes path —
                # ``model_name`` is the routed model from
                # get_baml_registry_for_user.
                effective_model_name=model_name,
                # P2 fix on commit 89aee97: thread the real provider
                # so cost reports show ollama:<model> for Ollama
                # rather than openai:<model> (which is what BAML's
                # openai-generic provider string would otherwise
                # infer).
                effective_provider=_provider_string_to_enum(provider),
            )
        else:
            tb = TypeBuilder()
            res = build_from_pydantic(response_model, tb)
            tb.DynamicContainer.add_property("data", res)
            result = b.ExtractRelationshipsFromChunk(
                chunk, context, {"tb": tb, "client_registry": client_registry}
            )
            result_model = response_model.model_validate(result.data)

        await _CHUNK_REL_CACHE.set(cache_key, result_model.model_dump(mode="json"))
        logger.debug(
            "BAML chunk relationship extraction response",
            extra={
                "transform_id": transform_id,
                "chunk_chars": len(chunk),
                "cached": False,
                "cache_key": cache_key,
            },
        )
        return result_model

    # @cached(ttl=86400,
    #     key_builder=lambda f, *args, **kwargs: f"{md5(kwargs['rel_type']+':'+kwargs['source_type']+':'+kwargs['source_entities']+':'+kwargs['target_type']+':'+kwargs['target_entities'])}")
    async def infer_relationship(
        self,
        rel_type: str,
        source_type: str = "",
        source_entities: str = "",
        target_type: str = "",
        target_entities: str = "",
        existing_rels: str = "",
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> List[RelationshipInference]:
        if not user_id:
            raise ValueError("user_id is required to get LLM credentials")

        # Get user's LLM credentials and create client registry
        client_registry, model_name, _provider = await get_baml_registry_for_user(
            user_id
        )

        # Use BAML tracking if user_id provided
        if user_id:
            from graphora_server.utils.baml_usage_tracker import (
                track_baml_infer_relationship,
            )

            return await track_baml_infer_relationship(
                user_id=user_id,
                rel_type=rel_type,
                source_type=source_type,
                source_entities=source_entities,
                target_type=target_type,
                target_entities=target_entities,
                existing_rels=existing_rels,
                transform_id=transform_id,
                document_usage_id=document_usage_id,
                client_registry=client_registry,
            )
        else:
            # Use dynamic client registry
            return b.InferRelationship(
                rel_type=rel_type,
                source_type=source_type,
                source_entities=source_entities,
                target_type=target_type,
                target_entities=target_entities,
                existing_rels=existing_rels,
                baml_options={"client_registry": client_registry},
            )

    # @cached(ttl=86400,
    #     key_builder=lambda f, *args, **kwargs: f"{md5(kwargs['entity_group_type']+':'+kwargs['entities_json'])}")
    async def standardise_properties(
        self,
        entity_group_type: str,
        entities_json: str,
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> List[StandardisedProperties]:
        if not user_id:
            raise ValueError("user_id is required to get LLM credentials")

        # Get user's LLM credentials and create client registry
        client_registry, model_name, _provider = await get_baml_registry_for_user(
            user_id
        )

        # Use BAML tracking if user_id provided
        if user_id:
            from graphora_server.utils.baml_usage_tracker import (
                track_baml_standardise_properties,
            )

            return await track_baml_standardise_properties(
                user_id=user_id,
                entity_group_type=entity_group_type,
                entities_json=entities_json,
                transform_id=transform_id,
                document_usage_id=document_usage_id,
                client_registry=client_registry,
            )
        else:
            # Use dynamic client registry
            return b.StandardiseProperties(
                entity_group_type=entity_group_type,
                entities_json=entities_json,
                baml_options={"client_registry": client_registry},
            )

    # @cached(ttl=86400,
    #     key_builder=lambda f, *args, **kwargs: f"{md5(kwargs['entity_type']+':'+kwargs['node_dicts_str'])}")
    async def resolve_entities(
        self,
        entity_type: str,
        node_dicts_str: str,
        user_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        document_usage_id: Optional[str] = None,
    ) -> List[ResolvedEntities]:
        if not user_id:
            raise ValueError("user_id is required to get LLM credentials")

        # Get user's LLM credentials and create client registry
        client_registry, model_name, _provider = await get_baml_registry_for_user(
            user_id
        )

        # Use BAML tracking if user_id provided
        if user_id:
            from graphora_server.utils.baml_usage_tracker import (
                track_baml_resolve_entities,
            )

            return await track_baml_resolve_entities(
                user_id=user_id,
                entity_type=entity_type,
                node_dicts_str=node_dicts_str,
                transform_id=transform_id,
                document_usage_id=document_usage_id,
                client_registry=client_registry,
            )
        else:
            # Use dynamic client registry
            return b.ResolveEntities(
                entity_type=entity_type,
                node_dicts_str=node_dicts_str,
                baml_options={"client_registry": client_registry},
            )
