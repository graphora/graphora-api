// BAML prompts for multi-pass extraction refinement
// Enhanced with chain-of-thought reasoning for improved refinement quality

// Refine an incomplete entity by finding missing properties
function RefineIncompleteEntity(
    chunk: string,
    entity_context: string,
    missing_properties: string,
    entity_type: string
) -> DynamicContainer {
    client DynamicGemini
    prompt #"
        You are refining an incomplete entity extraction. Focus on finding the missing information.

        Entity Type: {{entity_type}}

        Current Entity State:
        {{entity_context}}

        Missing or Uncertain Properties that need to be found:
        {{missing_properties}}

        Text to re-examine:
        ---
        {{chunk}}
        ---

        ## Chain-of-Thought Refinement Process

        Before extracting, follow these steps:

        1. **SCAN FOR EXPLICIT MENTIONS**: Look for direct statements of the missing properties.
           - Search for keywords related to each missing property
           - Check for alternative phrasings or synonyms

        2. **ANALYZE IMPLICIT INFORMATION**: Look for indirect clues:
           - Context surrounding the entity that implies property values
           - Related entities whose properties might inform this entity
           - Temporal or causal relationships that suggest values

        3. **CHECK FOR COREFERENCES**: The missing information might be:
           - Stated elsewhere in the text using pronouns or references
           - Mentioned in relation to another entity
           - Part of a list or enumeration

        4. **VERIFY CONSISTENCY**: Before including any value:
           - Ensure it doesn't conflict with existing properties
           - Confirm the value matches the expected property type
           - Consider if the value makes sense in context

        ## Instructions

        1. Focus ONLY on finding the missing properties listed above
        2. Only extract properties that are explicitly or implicitly present in the text
        3. If a property cannot be found with reasonable confidence, do not include it
        4. Maintain consistency with the existing entity information

        {{ ctx.output_format }}
    "#
}

// Find relationships for orphan nodes that have no connections
function FindMissingRelationships(
    chunk: string,
    orphan_nodes: string,
    existing_relationships: string,
    ontology_relationships: string
) -> DynamicContainer {
    client DynamicGemini
    prompt #"
        You are finding relationships for isolated entities that currently have no connections.

        These entities have NO relationships yet and need to be connected:
        {{orphan_nodes}}

        Existing relationships in the graph (for context):
        {{existing_relationships}}

        Possible relationship types from ontology:
        {{ontology_relationships}}

        Text to examine for relationships:
        ---
        {{chunk}}
        ---

        Instructions:
        1. Focus on finding relationships that connect the orphan nodes to other entities
        2. Look for both explicit and implicit relationships in the text
        3. Each relationship must have a valid source_id and target_id from existing entities
        4. Only extract relationships that are clearly supported by the text
        5. Use the relationship types defined in the ontology

        {{ ctx.output_format }}
    "#
}

// Re-extract entities with low confidence scores
function RefineUncertainEntities(
    chunk: string,
    uncertain_entities: string,
    ontology_context: string
) -> DynamicContainer {
    client DynamicGemini
    prompt #"
        You are verifying and improving uncertain entity extractions.

        These entities were extracted with LOW CONFIDENCE and need verification:
        {{uncertain_entities}}

        Ontology context (entity definitions):
        {{ontology_context}}

        Text to re-examine:
        ---
        {{chunk}}
        ---

        Instructions:
        1. Carefully verify each uncertain entity against the source text
        2. Correct any extraction errors found
        3. Fill in any missing properties that can now be found
        4. If an entity was incorrectly extracted, do not include it
        5. Increase confidence scores where evidence clearly supports the extraction
        6. Only include entities that are clearly supported by the text

        {{ ctx.output_format }}
    "#
}

// Extract specific relationship types that are expected but missing
function ExtractExpectedRelationships(
    chunk: string,
    source_entity: string,
    expected_relationship_type: string,
    target_entity_type: string,
    potential_targets: string
) -> DynamicContainer {
    client DynamicGemini
    prompt #"
        You are extracting a specific relationship that is expected but was not found.

        Source Entity:
        {{source_entity}}

        Expected Relationship Type: {{expected_relationship_type}}
        Target Entity Type: {{target_entity_type}}

        Potential target entities:
        {{potential_targets}}

        Text to examine:
        ---
        {{chunk}}
        ---

        Instructions:
        1. Find the {{expected_relationship_type}} relationship from the source entity
        2. The target must be one of the potential targets listed, or a new entity of type {{target_entity_type}}
        3. Look for both explicit mentions and implicit connections
        4. Only extract if the relationship is clearly supported by the text
        5. Include any relationship properties if present

        {{ ctx.output_format }}
    "#
}

// Comprehensive entity and relationship refinement in one pass
// Enhanced with structured chain-of-thought reasoning
function ComprehensiveRefinement(
    chunk: string,
    gaps_context: string,
    existing_entities: string,
    existing_relationships: string,
    ontology_yaml: string
) -> DynamicContainer {
    client DynamicGemini
    prompt #"
        You are performing a comprehensive refinement pass to address extraction gaps.

        ## EXTRACTION GAPS TO ADDRESS

        {{gaps_context}}

        ## EXISTING ENTITIES (for reference)

        {{existing_entities}}

        ## EXISTING RELATIONSHIPS (for reference)

        {{existing_relationships}}

        ## ONTOLOGY DEFINITION

        {{ontology_yaml}}

        ## TEXT TO RE-EXAMINE

        ---
        {{chunk}}
        ---

        ## Chain-of-Thought Refinement Process

        Work through each gap type systematically:

        ### STEP 1: Incomplete Entities
        For each incomplete entity:
        a) Identify the specific missing properties
        b) Search the text for explicit mentions of those properties
        c) Look for implicit information (titles, roles, descriptions that imply values)
        d) Check if information appears in relation to other entities
        e) Only include values you can justify from the text

        ### STEP 2: Orphan Nodes
        For each orphan node:
        a) Review the ontology to identify possible relationship types
        b) Search for verbs, actions, or associations involving this entity
        c) Check for membership/ownership/location relationships
        d) Look for transitive relationships (A relates to B, B relates to C)
        e) Connect to existing entities using their IDs

        ### STEP 3: Low Confidence Extractions
        For each low-confidence entity:
        a) Re-read the relevant text sections carefully
        b) Verify the entity type is correct
        c) Confirm property values against the source text
        d) Look for additional evidence that increases confidence
        e) Correct any errors found

        ### STEP 4: Missing Relationships
        For expected but missing relationships:
        a) Identify source and target entity types
        b) Search for relationship indicators (verbs, prepositions, possessives)
        c) Check for implicit connections through shared context
        d) Verify both source and target entities exist
        e) Use existing entity IDs for all references

        ## Output Rules

        - Only extract information explicitly present in the text
        - Maintain consistency with existing entities and relationships
        - Do not duplicate existing entities - use their IDs for relationships
        - Include confidence scores reflecting certainty:
          * 0.9-1.0: Explicitly stated
          * 0.7-0.9: Clearly implied
          * 0.5-0.7: Reasonably inferred
        - If no new information can be found, return minimal output

        {{ ctx.output_format }}
    "#
}
