class ResolvedEntities {
  matching_ids string[] @description(#"
    IDs of entities that match.
    E.g: ['2e89f864-0e7e', '572077c6-a044', '587a-bfb7-4ec535f1a005'],  # Example: entities with IDs '2e89f864-0e7e', '572077c6-a044', '587a-bfb7-4ec535f1a005' match
  "#)
  confidence_score float @description(#"
    Confidence score the matching group.
    E.g: 0.95
  "#)
  explanation string @description(#"
    Explanations for each matching group.
    E.g: These entities share the same name with minor variations and have overlapping properties
  "#)
}


function ResolveEntities(entity_type: string, node_dicts_str: string) -> ResolvedEntities[] {
  client DynamicGemini
  prompt #"
    You are performing entity resolution on a group of {{entity_type}} entities.
    Your task is to identify which entities refer to the same real-world entity and should be merged.
    
    Entities:
    {{ node_dicts_str }}
    
    Use these guidelines for matching:
    1. Compare all properties to determine if entities match
    2. Handle variations in naming, formatting, and completeness
    3. Consider similarity in key identifying properties
    4. Be conservative - only match if reasonably confident (>80% sure)
    5. Consider property values that are similar but not exactly matching
    6. Look for complementary information across entities

    {{ ctx.output_format }}
  "#
}