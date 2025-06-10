// Defining a data model.
class DynamicContainer {
  @@dynamic
}

// Create a function to extract the nodes & edges from a string.
function ExtractNodesFromChunk(chunk: string, context: string) -> DynamicContainer {
  client DynamicGemini
  prompt #"
    Extract structured information representing Entities from the text chunk according to the ontology specification.

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
    {{context}}
    ```

    Chunk to process:
    {{ chunk }}

    Remember:
    - Only extract information that is explicitly present in the text
    - Set confidence scores based on certainty of extraction
    - Include all required fields for each node
    - Omit optional fields if information is not clearly present
    

    {{ ctx.output_format }}
  "#
}

function ExtractRelationshipsFromChunk(chunk: string, context: string) -> DynamicContainer {
  client DynamicGemini
  prompt #"
    Extract structured information representing relationships from the text chunk according to the ontology specification.

    These nodes & relationships were identified from the previous text chunks of the same doc.
    Format: (:source_node_type {node_id & properties}) - [:relationship_type {properties}] -> (:target_node_type {node_id & properties})
    ```
    {{context}}
    ```

    Output a JSON object following these rules:
    1. "<source>_<relationship>_<target>" for each relationship (e.g., "Company_HAS_BUSINESS_Business").
    2. Each relationship MUST include:
       - "source_id": The ID of the source node from the provided entities.
       - "target_id": The ID of the target node from the provided entities.
       - "properties": Any additional relationship properties (optional).
    3. Metadata: "extraction_timestamp" (ISO), "tokens_used", "confidence_score" (0.0-1.0).

    Chunk to process:
    {{ chunk }}

    Remember:
    - Only extract information that is explicitly present in the text
    - Set confidence scores based on certainty of extraction
    - Include all required fields for each relationship
    - Omit optional fields if information is not clearly present
    

    {{ ctx.output_format }}
  "#
}

