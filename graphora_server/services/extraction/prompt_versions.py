"""B0-prov-extend: BAML prompt version registry.

Maps each BAML function to a semantic version string. Stamped onto
``NodeProvenance.prompt_version`` (and onto node/edge properties via
``_attach_provenance_properties``) so a user inspecting an extracted
fact can correlate it with a specific prompt iteration.

The version is a static constant per function rather than a runtime
hash of the BAML source. Hashes change every time someone edits a
prompt during iteration; semantic versions only change when the
behavioural contract changes. Citations should be stable across
small wording tweaks.

**Maintenance contract:** when a BAML function's prompt is changed
in a way that affects extraction behaviour (different system message,
different output schema, different few-shot examples), bump the
version here. When the change is comment-only or whitespace-only,
leave the version. The harm of a stale version is "wrong version
label," not broken extraction — but the version field is meaningless
if it never changes when the prompts do.

Bump conventions:
    * Patch (1.0.0 → 1.0.1): wording polish that doesn't change
      observable output (clarification phrases, typo fixes)
    * Minor (1.0.0 → 1.1.0): output shape additions or backward-
      compatible behavioural shifts (new output field, new few-shot
      example category)
    * Major (1.0.0 → 2.0.0): output shape changes, prompt structure
      changes, model assumptions shift

CI doesn't enforce these — the registry is documentation by
convention.
"""

from typing import Optional


# Initial registry — all functions started at v1.0.0 for the B0
# landing. Bumped to v1.1.0 in commit d928586 (Gate 4 per-fact
# source_excerpt) — the four extraction functions added a new
# optional output field (source_excerpt per entity / relationship)
# AND new prompt sections instructing the model to emit it. Per the
# minor-bump convention above (output shape additions), the version
# moves. The cache-key layer in services/llm/client.py threads this
# value into every cache key so a v1.0.0 cached response doesn't
# serve a v1.1.0 caller and silently skip Gate 4.
BAML_PROMPT_VERSIONS = {
    "ExtractNodesFromChunk": "v1.1.0",
    "ExtractRelationshipsFromChunk": "v1.1.0",
    "ExtractNodesFromPdf": "v1.1.0",
    "ExtractRelationshipsFromPdf": "v1.1.0",
}


def get_prompt_version(baml_function_name: str) -> Optional[str]:
    """Return the semantic version for a BAML function.

    Returns ``None`` for functions not in the registry — callers
    should treat that as "version unknown" rather than an error so
    new BAML functions don't crash extraction before getting their
    first registry entry.
    """
    return BAML_PROMPT_VERSIONS.get(baml_function_name)
