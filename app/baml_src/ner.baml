// Defining a data model.
class DynamicContainer {
  @@dynamic
}

// Create a function to extract the nodes & edges from a string.
function ExtractChunk(chunk: string, context: string) -> DynamicContainer {
  client VertexAI
  prompt #"
    Extract structured information from the text chunk according to the ontology specification.

    Format the output as a JSON object with the following structure:
    1. For each entity type, include a list field named "<entity_type>_list" containing all instances
    2. For each relationship type, include a list field named "<source>_<relationship>_<target>" containing all instances
    3. Include metadata fields:
      - extraction_timestamp: Current timestamp in ISOformat string
      - tokens_used: Number of tokens used (if available)
      - confidence_score: Overall confidence in extraction (0.0 to 1.0)
    4. Omit optional fields if information is not clearly present
    5. No additional properties. Just the specified fields.

    When extracting new information, maintain consistency with these previously identified entities. 
    These were identified from the previous text chunks of the same doc.
    ```
    {{context}}
    ```

    Chunk to process:
    {{ chunk }}

    Remember:
    - Only extract information that is explicitly present in the text
    - Set confidence scores based on certainty of extraction
    - Include all required fields for each entity/relationship
    - Omit optional fields if information is not clearly present
    

    {{ ctx.output_format }}
  "#
}

