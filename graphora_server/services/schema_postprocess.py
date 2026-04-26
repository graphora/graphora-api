"""Post-hoc ontology inference from an extracted graph.

Where ``schema_inference.py`` peeks at raw document text BEFORE
extraction and asks the LLM "what schema should we use?",
``schema_postprocess`` runs AFTER extraction and asks the LLM
"given what was actually extracted, what's the ontology that
best describes it?"

Why split them:
    * Pre-extraction inference biases the extractor — the LLM
      commits to categories it hasn't seen evidence for yet.
    * Post-hoc inference lets the document speak first. The
      types that *emerged* from extraction are the input
      signal; the LLM's job is to cluster, rename, and
      disambiguate rather than invent.

The output is a YAML ontology in the same shape as
``SCHEMA_INFERENCE_PROMPT``'s so downstream consumers (validators,
ontology storage, the visual editor) stay unchanged.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List

import yaml
from google.genai import types

from graphora_server.utils.llm_helper import (
    get_llm_client_for_user,
    get_user_llm_credentials,
)

logger = logging.getLogger(__name__)


_MAX_NODE_SAMPLES_PER_TYPE = 5
_MAX_EDGE_SAMPLES_PER_TYPE = 10
_MAX_PROMPT_CHARS = 16000


POSTPROCESS_PROMPT = """You are refining a knowledge graph ontology BASED ON WHAT WAS EXTRACTED.

The extraction pipeline has already produced a graph. Below is a summary
of the node types, property shapes, and relationship patterns that actually
emerged. Your job is to produce a refined ontology that describes this
graph accurately — cluster near-duplicate types, standardize naming, and
clarify relationship semantics.

<graph_summary>
{graph_summary}
</graph_summary>

Rules:
- The ontology MUST cover every emerged type — do not drop types.
- You MAY merge types that are obvious duplicates (e.g. "Person"/"Individual"
  → "Person"). When you merge, keep the more standard/common name.
- Use PascalCase for entity names, SCREAMING_SNAKE_CASE for relationships.
- Include 2-5 properties per entity, drawn from the observed property names.
- Mark identifying properties (name, id, title) as required when present.
- Do not invent relationship types the data does not support.

Output ONLY valid YAML in this exact format (no markdown code blocks):

version: "0.1.0"
entities:
  EntityName:
    description: "Brief description of the entity"
    properties:
      property_name:
        type: str
        description: "Property description"
        required: true
relationships:
  RELATIONSHIP_TYPE:
    description: "Brief description"
    source: SourceEntity
    target: TargetEntity
    properties: {{}}
"""


def summarize_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> str:
    """Produce a compact, LLM-friendly summary of what was extracted.

    Avoids dumping the full graph — large extractions would blow the
    prompt budget and drown the signal in noise. Instead:

    * Group nodes by type, show type counts and the union of property
      keys seen, plus a handful of example ``label`` values per type.
    * Group edges by ``(source_type, rel_type, target_type)``; list
      the most frequent triples.

    The resulting summary is bounded by constants at module scope —
    never exceeds ``_MAX_PROMPT_CHARS`` even on extractions with
    thousands of nodes.
    """
    type_to_nodes: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        t = str(n.get("type") or "Unknown")
        type_to_nodes.setdefault(t, []).append(n)

    # Build node-type profiles.
    type_profiles: List[str] = []
    for t, bucket in sorted(type_to_nodes.items()):
        prop_counter: Counter[str] = Counter()
        sample_labels: List[str] = []
        for node in bucket:
            props = node.get("properties") or {}
            prop_counter.update(props.keys())
            label = node.get("label") or props.get("name") or props.get("title")
            if label and len(sample_labels) < _MAX_NODE_SAMPLES_PER_TYPE:
                sample_labels.append(str(label)[:60])
        top_props = [f"{name}({count})" for name, count in prop_counter.most_common(6)]
        examples = ", ".join(sample_labels) if sample_labels else "(no labels)"
        type_profiles.append(
            f"- {t} ({len(bucket)} nodes) | props: {', '.join(top_props) or '(none)'} "
            f"| examples: {examples}"
        )

    # Build edge-pattern profiles: (src_type, rel_type, tgt_type) -> count.
    id_to_type = {n.get("id"): n.get("type") for n in nodes if "id" in n}
    triple_counter: Counter[tuple] = Counter()
    for e in edges:
        src_t = id_to_type.get(e.get("source"), "Unknown")
        tgt_t = id_to_type.get(e.get("target"), "Unknown")
        rel_t = str(e.get("type") or "RELATED_TO")
        triple_counter[(src_t, rel_t, tgt_t)] += 1

    edge_lines: List[str] = []
    for (src_t, rel_t, tgt_t), count in triple_counter.most_common(
        _MAX_EDGE_SAMPLES_PER_TYPE
    ):
        edge_lines.append(f"- {src_t} --{rel_t}-> {tgt_t} ({count})")

    summary_parts = [
        f"Entity types ({len(type_to_nodes)}):",
        *type_profiles,
        "",
        f"Relationship patterns ({len(triple_counter)} distinct, showing top "
        f"{min(len(triple_counter), _MAX_EDGE_SAMPLES_PER_TYPE)}):",
        *edge_lines,
    ]
    full = "\n".join(summary_parts)
    return full[:_MAX_PROMPT_CHARS]


def _parse_yaml_response(response_text: str) -> Dict[str, Any]:
    """Parse YAML from an LLM response, tolerating code fences."""
    yaml_match = re.search(r"```(?:yaml)?\n(.*?)```", response_text, re.DOTALL)
    yaml_content = yaml_match.group(1) if yaml_match else response_text
    yaml_content = yaml_content.strip()

    try:
        parsed = yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        logger.error("Failed to parse post-hoc YAML: %s", exc)
        logger.debug("Raw YAML content: %s", yaml_content)
        raise ValueError(f"Failed to parse inferred ontology: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected dict ontology, got {type(parsed).__name__}")
    return parsed


async def infer_ontology_from_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    user_id: str,
    *,
    client_factory=None,
) -> Dict[str, Any]:
    """Infer a refined ontology YAML dict from an emerged graph.

    Provider-agnostic: uses ``get_llm_client_for_user`` so the same
    code path runs against Gemini (default) or Ollama (when
    LLM_PROVIDER=ollama is set or the user's stored provider is
    ollama). Both clients expose the same
    ``client.models.generate_content`` shape.

    Args:
        nodes: Extracted nodes as dicts with at least ``id``, ``type``,
            optionally ``label`` and ``properties``.
        edges: Extracted edges as dicts with ``source``, ``target``,
            ``type``.
        user_id: For LLM credential lookup.
        client_factory: Test hook. Legacy 2-tuple form
            ``(api_key) -> client`` is still accepted for backward
            compat with the existing tests; when set, the legacy
            Gemini-only path is taken.

    Returns:
        Parsed ontology dict with keys ``version``, ``entities``, and
        ``relationships``. ``relationships`` may be an empty dict.

    Raises:
        ValueError: If the graph is empty or the LLM returns
            unparseable YAML.
        NoAIConfigurationError: If the user has no LLM configured.
    """
    if not nodes:
        raise ValueError("Cannot infer ontology from an empty graph")

    summary = summarize_graph(nodes, edges)
    prompt = POSTPROCESS_PROMPT.format(graph_summary=summary)

    if client_factory is not None:
        # Legacy test-hook path. Keeps the existing test suite green
        # while the new provider abstraction lands.
        api_key, model_name = await get_user_llm_credentials(user_id)
        client = client_factory(api_key)
    else:
        client, model_name, _provider = await get_llm_client_for_user(user_id)

    logger.info(
        "Running post-hoc ontology inference for user %s over %d nodes / %d edges",
        user_id,
        len(nodes),
        len(edges),
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=4096,
        ),
    )
    response_text = (response.text or "").strip()
    if not response_text:
        raise ValueError("Empty response from LLM during post-hoc inference")

    ontology = _parse_yaml_response(response_text)

    if "entities" not in ontology or not ontology["entities"]:
        raise ValueError("Inferred ontology has no entities")
    ontology.setdefault("version", "0.1.0")
    ontology.setdefault("relationships", {})

    # Drop non-string property-type leaf-values that sneak in when the
    # LLM confuses "type" (data type) with "type" (entity type).
    for entity_name, entity_def in (ontology.get("entities") or {}).items():
        props = (entity_def or {}).get("properties") or {}
        for prop_name, prop_def in props.items():
            if isinstance(prop_def, dict):
                raw_type = prop_def.get("type")
                if not isinstance(raw_type, str):
                    prop_def["type"] = "str"
                    logger.debug(
                        "Coerced %s.%s property type to str (was %r)",
                        entity_name,
                        prop_name,
                        raw_type,
                    )

    logger.info(
        "Post-hoc inference produced %d entities and %d relationships",
        len(ontology.get("entities") or {}),
        len(ontology.get("relationships") or {}),
    )
    return ontology


def ontology_dict_to_yaml(ontology: Dict[str, Any]) -> str:
    """Render an ontology dict to the canonical YAML shape on disk.

    Kept as a module-level helper so API handlers and Phase 2 pipeline
    callers share a single representation.
    """
    return yaml.dump(ontology, default_flow_style=False, sort_keys=False)
