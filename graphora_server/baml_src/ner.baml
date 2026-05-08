// Defining a data model.
class DynamicContainer {
  @@dynamic
}

// Create a function to extract the nodes & edges from a string.
// Enhanced with chain-of-thought reasoning for improved extraction quality.
function ExtractNodesFromChunk(chunk: string, context: string) -> DynamicContainer {
  client DynamicGemini
  prompt #"
    Extract structured information representing Entities from the text chunk according to the ontology specification.

    ## Chain-of-Thought Extraction Process

    Before extracting, follow these reasoning steps:

    1. **IDENTIFY ENTITY MENTIONS**: Read through the text and identify all potential entity mentions.
       - Look for named entities (people, organizations, places, products, etc.)
       - Look for references to concepts defined in the ontology
       - Note any pronouns or references that may refer to existing entities

    2. **RESOLVE REFERENCES**: For each mention, determine if it refers to:
       - A new entity (create new entry)
       - An existing entity from previous context (use existing ID)
       - An entity mentioned earlier in this same chunk (deduplicate)

    3. **EXTRACT PROPERTIES**: For each entity:
       - Identify all properties mentioned in the text
       - Consider implicit properties (e.g., "the CEO" implies role="CEO")
       - Assess confidence for each property value

    4. **CONSIDER RELATIONSHIPS**: While extracting entities, note:
       - What relationships this entity might have with others
       - Missing entities that relationships imply should exist
       - This helps ensure complete entity extraction

    ## Output Format

    Format the output as a JSON object with the following structure:
    1. For each entity type, include a list field named "<entity_type>_list" containing all instances
    2. Include metadata fields:
      - extraction_timestamp: Current timestamp in ISOformat string
      - tokens_used: Number of tokens used (if available)
      - confidence_score: Overall confidence in extraction (0.0 to 1.0)
    3. Omit optional fields if information is not clearly present
    4. No additional properties. Just the specified fields.

    ## Previously Identified Entities

    When extracting new information, maintain consistency with these previously identified entities.
    If previous Entities miss any properties add them.
    Use the `id` field to refer & match the nodes below.
    ```
    {{context}}
    ```

    ## Text Chunk to Process

    {{ chunk }}

    ## Extraction Guidelines

    - Only extract information that is explicitly present in the text
    - Set confidence scores based on certainty of extraction:
      * 0.9-1.0: Explicitly stated, unambiguous
      * 0.7-0.9: Clearly implied or strongly inferred
      * 0.5-0.7: Reasonably inferred but some ambiguity
      * Below 0.5: Uncertain, may need verification
    - Include all required fields for each node
    - Omit optional fields if information is not clearly present
    - When in doubt about entity boundaries, prefer more specific entities

    ## Per-Entity Source Excerpt

    For EACH extracted entity, include a `source_excerpt` field — a
    1-2 sentence VERBATIM quote from the text chunk where this
    entity is mentioned. This powers the Evidence-tab provenance
    surface for end users.

    Rules:
    - Use the EXACT WORDS from the source text, not a paraphrase.
    - Choose the most informative single sentence (or two adjacent
      sentences) that mentions the entity.
    - Keep it short — ideally under 200 characters.
    - If the entity is mentioned multiple times, pick the mention
      that includes the most identifying information (full name,
      role, etc.).
    - If you cannot identify a specific source sentence, omit the
      field rather than fabricating one.


    {{ ctx.output_format }}
  "#
}

// Enhanced relationship extraction with chain-of-thought reasoning
function ExtractRelationshipsFromChunk(chunk: string, context: string) -> DynamicContainer {
  client DynamicGemini
  prompt #"
    Extract structured information representing relationships from the text chunk according to the ontology specification.

    ## Chain-of-Thought Relationship Extraction

    Before extracting, follow these reasoning steps:

    1. **IDENTIFY RELATIONSHIP SIGNALS**: Scan the text for indicators of relationships:
       - Verbs that indicate actions or states (e.g., "works for", "created", "manages")
       - Possessives (e.g., "company's CEO", "team members")
       - Prepositions indicating associations (e.g., "part of", "located in", "affiliated with")
       - Temporal or causal connections

    2. **MAP TO ENTITY PAIRS**: For each relationship signal:
       - Identify the source entity (who/what is performing or owning)
       - Identify the target entity (who/what is receiving or being owned)
       - Match these to entities in the provided context using their IDs

    3. **CLASSIFY RELATIONSHIP TYPE**: Determine the relationship type that best fits:
       - Use the ontology-defined relationship types when possible
       - Consider the directionality (source -> target)
       - Note any relationship properties mentioned

    4. **CHECK FOR IMPLICIT RELATIONSHIPS**: Look for relationships that are implied but not explicitly stated:
       - Organizational hierarchies (manager-report, parent-subsidiary)
       - Membership or participation (employee-company, author-publication)
       - Location or containment (city-country, department-organization)

    5. **CONNECT ORPHAN NODES**: Pay special attention to entities marked as "WITHOUT RELATIONSHIPS":
       - These need connections - look harder for relationships involving them
       - Consider transitive relationships through other entities

    ## Previously Identified Entities and Relationships

    These nodes & relationships were identified from the previous text chunks of the same doc.
    Format: (:source_node_type {node_id & properties}) - [:relationship_type {properties}] -> (:target_node_type {node_id & properties})
    ```
    {{context}}
    ```

    ## Output Format

    Output a JSON object following these rules:
    1. "<source>_<relationship>_<target>" for each relationship (e.g., "Company_HAS_BUSINESS_Business").
    2. Each relationship MUST include:
       - "source_id": The ID of the source node from the provided entities.
       - "target_id": The ID of the target node from the provided entities.
       - "properties": Any additional relationship properties (optional).
    3. Metadata: "extraction_timestamp" (ISO), "tokens_used", "confidence_score" (0.0-1.0).

    ## Text Chunk to Process

    {{ chunk }}

    ## Extraction Guidelines

    - Only extract relationships that are clearly supported by the text
    - Both source_id and target_id MUST reference existing entities from the context
    - Set confidence scores based on how explicitly the relationship is stated:
      * 0.9-1.0: Explicitly stated relationship
      * 0.7-0.9: Strongly implied relationship
      * 0.5-0.7: Reasonably inferred relationship
    - Do not create relationships where the connection is purely speculative
    - Prefer ontology-defined relationship types over creating new ones

    ## Per-Relationship Source Excerpt

    For EACH extracted relationship, include a `source_excerpt`
    field — a 1-2 sentence VERBATIM quote from the text chunk where
    the relationship is described. This powers the Evidence-tab
    provenance surface for end users.

    Rules:
    - Use the EXACT WORDS from the source text, not a paraphrase.
    - Choose the sentence that most clearly establishes the
      relationship between the source and target entities.
    - Keep it short — ideally under 200 characters.
    - If the relationship is implied across multiple sentences,
      pick the one that most directly connects the two entities.
    - If you cannot identify a specific source sentence, omit the
      field rather than fabricating one.


    {{ ctx.output_format }}
  "#
}

