// BAML prompts for multi-pass extraction refinement

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

        Instructions:
        1. Focus ONLY on finding the missing properties listed above
        2. Only extract properties that are explicitly present in the text
        3. Look for implicit information that may provide the missing values
        4. If a property cannot be found, do not make up values
        5. Maintain consistency with the existing entity information

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

        GAPS TO ADDRESS:
        {{gaps_context}}

        EXISTING ENTITIES (for reference):
        {{existing_entities}}

        EXISTING RELATIONSHIPS (for reference):
        {{existing_relationships}}

        ONTOLOGY:
        {{ontology_yaml}}

        TEXT TO RE-EXAMINE:
        ---
        {{chunk}}
        ---

        Instructions:
        1. Address each gap listed above systematically
        2. For incomplete entities: find missing property values
        3. For orphan nodes: find relationships connecting them
        4. For low confidence: verify and correct extractions
        5. For missing relationships: extract expected connections

        Rules:
        - Only extract information explicitly present in the text
        - Maintain consistency with existing entities and relationships
        - Do not duplicate existing entities - use their IDs for relationships
        - Include confidence scores reflecting certainty of extraction

        {{ ctx.output_format }}
    "#
}
