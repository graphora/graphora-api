// Models for conflict analysis pipeline
type JsonValue = int | string | bool | float | JsonObject | JsonArray
type JsonObject = map<string, JsonValue>
type JsonArray = JsonValue[]

enum ResolutionStrategy {
  KEEP_STAGING
  KEEP_PRODUCTION
  MERGE_VALUES
  KEEP_BOTH
}

// Output types
class ConflictAnalysis {
  description string @description(#"
    Detailed description of the conflict.
    E.g: 'The conflict shows a difference in the age property between staging (35) and production (30)'
  "#)
  potential_causes string[] @description(#"
    List of potential causes for the conflict.
    E.g: ['Data entry error', 'Outdated information in production', 'Intentional update']
  "#)
  semantic_meaning string @description(#"
    Semantic interpretation of the differences.
    E.g: 'Age has been updated to reflect current information'
  "#)
  impact string @description(#"
    Business impact of the conflict.
    E.g: 'Minor impact - age difference is within reasonable range'
  "#)
  key_differences string[] @description(#"
    List of key fields that are different.
    E.g: ['age: 35 vs 30', 'last_updated: 2025-01-01 vs 2024-12-31']
  "#)
}

class ConflictClassification {
  category string @description(#"
    Classification category of the conflict.
    Must be one of: SIMPLE_PROPERTY, COMPLEX_PROPERTY, ENTITY_DUPLICATION, 
    MISSING_RELATIONSHIP, CONFLICTING_RELATIONSHIP, SCHEMA_CONFLICT, OTHER
  "#)
  explanation string @description(#"
    Explanation for the classification.
    E.g: 'This is a simple property conflict as only the age value differs'
  "#)
  confidence float @description(#"
    Confidence score for the classification (0-1).
    E.g: 0.95
  "#)
}

class PropertyConflictAnalysis {
  recommended_strategy ResolutionStrategy @description(#"
    Recommended resolution strategy
  "#)
  confidence float @description(#"
    Confidence score for the recommendation (0-1).
    E.g: 0.95
  "#)
  explanation string @description(#"
    Explanation of the analysis and recommendation.
    E.g: 'The staging value contains more detailed information while preserving key data'
  "#)
  can_auto_resolve bool @description(#"
    Whether this conflict can be safely auto-resolved.
    E.g: true
  "#)
  potential_risks string[] @description(#"
    List of potential risks or considerations.
    E.g: ['Data loss risk: production value contains unique identifiers']
  "#)
}

class ConflictGroupAnalysis {
  pattern string @description(#"
    Common pattern identified in the conflicts.
    E.g: 'All conflicts show systematic data format differences'
  "#)
  batch_resolvable bool @description(#"
    Whether these conflicts can be resolved as a batch.
    E.g: true
  "#)
  recommended_strategy ResolutionStrategy @description(#"
    Recommended batch resolution strategy
  "#)
  confidence float @description(#"
    Confidence in the batch resolution recommendation (0-1).
    E.g: 0.85
  "#)
  risks string[] @description(#"
    Potential risks of batch resolution.
    E.g: ['Some entities may require manual review']
  "#)
}

class RelationshipConflictAnalysis {
  recommended_strategy ResolutionStrategy @description(#"
    Recommended resolution strategy
  "#)
  confidence float @description(#"
    Confidence score for the recommendation (0-1).
    E.g: 0.95
  "#)
  explanation string @description(#"
    Explanation of the analysis and recommendation.
    E.g: 'The staging relationship has more complete metadata'
  "#)
  can_auto_resolve bool @description(#"
    Whether this conflict can be safely auto-resolved.
    E.g: true
  "#)
  semantic_impact string @description(#"
    Impact on graph semantics if resolved this way.
    E.g: 'May affect downstream analytics that rely on relationship direction'
  "#)
  risks string[] @description(#"
    Potential risks of the resolution.
    E.g: ['Data loss in relationship properties', 'Breaking existing queries']
  "#)
}

// Functions for conflict analysis pipeline
function AnalyzeConflict(
  entity_type: string,
  property_name: string,
  staging_value: string,
  production_value: string,
  value_type: string
) -> ConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze the following property conflict between staging and production values.
    Consider the context, data types, and domain knowledge to suggest the best resolution.

    Entity Type: {{entity_type}}
    Property Name: {{property_name}}
    Staging Value: {{staging_value}}
    Production Value: {{production_value}}
    Value Type: {{value_type}}

    Provide:
    1. Analysis of the difference
    2. Potential impact of each value
    3. Recommended resolution strategy
    4. Confidence in the recommendation (0-1)
    5. Whether this can be auto-resolved

    {{ ctx.output_format }}
  "#
}

function ClassifyConflict(
  analysis: string,
  conflict_type: string
) -> ConflictClassification {
  client DynamicGemini
  prompt #"
    Based on this conflict analysis:
    {{analysis}}

    Conflict Type: {{conflict_type}}

    Classify this conflict into one of these categories:
    1. SIMPLE_PROPERTY - Simple property value difference
    2. COMPLEX_PROPERTY - Complex structural property difference
    3. ENTITY_DUPLICATION - Potentially same entity represented differently
    4. MISSING_RELATIONSHIP - Missing relationship between entities
    5. CONFLICTING_RELATIONSHIP - Contradicting relationships
    6. SCHEMA_CONFLICT - Ontology/schema mismatch
    7. OTHER - Other conflict type

    Provide the classification category, a brief explanation, and your confidence level.

    {{ ctx.output_format }}
  "#
}

function GenerateResolutionOptions(
  conflict_analysis: string,
  classification: string,
  staging_element: string,
  prod_element: string
) -> ResolutionOption[] {
  client DynamicGemini
  prompt #"
    For this conflict:

    Analysis:
    {{conflict_analysis}}

    Classification:
    {{classification}}

    Staging Element:
    {{staging_element}}

    Production Element:
    {{prod_element}}

    Generate 3-5 distinct resolution options. For each option, provide:
    1. A short, descriptive name for the approach
    2. A detailed explanation of the resolution steps
    3. The business impact of this resolution
    4. Any risks or considerations

    All options should be valid solutions with different trade-offs.
    Be specific and actionable in your suggestions.

    {{ ctx.output_format }}
  "#
}

function AssessResolutionComplexity(
  conflict_type: string,
  classification: string,
  resolution_option: string
) -> float {
  client DynamicGemini
  prompt #"
    Assess the complexity of implementing this resolution:

    Conflict Type: {{conflict_type}}
    Classification: {{classification}}
    Resolution Option: {{resolution_option}}

    Consider these factors:
    1. Technical complexity of implementation
    2. Risk of data loss or corruption
    3. Need for human intervention
    4. Impact on related data
    5. Rollback complexity

    Provide a complexity score between 0 (very simple) and 1 (very complex).
    
    Example scores:
    - 0.2: Simple property update, no dependencies
    - 0.5: Moderate complexity, some validation needed
    - 0.8: Complex merge operation, multiple dependencies
    - 1.0: Very complex, high risk, requires careful planning

    {{ ctx.output_format }}
  "#
}

function AnalyzePropertyConflict(
  entity_type: string,
  property_name: string,
  staging_value: string,
  production_value: string,
  value_type: string,
  historical_resolutions: string
) -> PropertyConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this property conflict between staging and production values:

    Entity Type: {{entity_type}}
    Property Name: {{property_name}}
    Value Type: {{value_type}}
    
    Staging Value: {{staging_value}}
    Production Value: {{production_value}}
    
    Historical Resolutions:
    {{historical_resolutions}}
    
    Guidelines:
    1. Consider the property semantics and entity type
    2. Evaluate data quality and completeness
    3. Consider historical resolution patterns
    4. Assess risks of data loss
    5. Determine if auto-resolution is safe
    
    {{ ctx.output_format }}
  "#
}

function AnalyzeConflictGroup(
  entity_type: string,
  property_name: string,
  value_type: string,
  conflict_count: int,
  sample_conflicts: string
) -> ConflictGroupAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this group of similar conflicts to determine if they can be batch resolved:

    Entity Type: {{entity_type}}
    Property Name: {{property_name}}
    Value Type: {{value_type}}
    Total Conflicts: {{conflict_count}}
    
    Sample Conflicts:
    {{sample_conflicts}}
    
    Guidelines:
    1. Look for systematic patterns in the conflicts
    2. Assess if a single strategy can safely apply to all
    3. Consider edge cases needing manual review
    4. Evaluate risks of batch resolution
    
    {{ ctx.output_format }}
  "#
}

function AnalyzeRelationshipConflict(
  relationship_type: string,
  conflict_type: string,
  staging_details: string,
  production_details: string,
  graph_context: string
) -> RelationshipConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this relationship conflict between staging and production:

    Relationship Type: {{relationship_type}}
    Conflict Type: {{conflict_type}}
    
    Staging Details:
    {{staging_details}}
    
    Production Details:
    {{production_details}}
    
    Graph Context:
    {{graph_context}}
    
    Guidelines:
    1. Consider the relationship semantics and direction
    2. Evaluate impact on graph connectivity
    3. Assess risks to data integrity
    4. Consider query patterns that may be affected
    5. Evaluate if auto-resolution is safe
    
    {{ ctx.output_format }}
  "#
}

function AnalyzePropertyConflictWithOntology(
  conflict: JsonObject,
  ontology: JsonObject
) -> ConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze the following property conflict between staging and production values.
    
    ## Conflict Details
    {{conflict}}
    
    ## Ontology Context
    {{ontology}}
    
    Provide a detailed analysis of this property conflict, including:
    1. A clear description of what's different
    2. Potential causes for this difference
    3. The semantic meaning of the change
    4. Business impact of this conflict
    5. Key differences in a structured format
    
    Focus on understanding the nature of the property conflict and its implications.
  "#
}

function AnalyzeRelationshipConflictWithOntology(
  conflict: JsonObject,
  ontology: JsonObject
) -> ConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze the following relationship conflict between staging and production.
    
    ## Conflict Details
    {{conflict}}
    
    ## Ontology Context
    {{ontology}}
    
    Provide a detailed analysis of this relationship conflict, including:
    1. A clear description of what's different in the relationship
    2. Potential causes for this difference
    3. The semantic meaning of the change in relationship
    4. Business impact of this relationship conflict
    5. Key differences in a structured format
    
    Focus on understanding how this relationship change affects the connected entities and overall data integrity.
  "#
}

function AnalyzeEntityMatchConflictWithOntology(
  conflict: JsonObject,
  ontology: JsonObject
) -> ConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze the following entity matching conflict where a staging entity matches multiple production entities.
    
    ## Conflict Details
    {{conflict}}
    
    ## Ontology Context
    {{ontology}}
    
    Provide a detailed analysis of this entity matching conflict, including:
    1. A clear description of why multiple matches were found
    2. Potential causes for these duplicate entities
    3. The semantic meaning of these matches
    4. Business impact of having duplicate entities
    5. Key similarities and differences between the matching entities
    
    Focus on understanding why these entities might be duplicates and the implications of merging or keeping them separate.
  "#
}

function AnalyzeGenericConflictWithOntology(
  conflict: JsonObject,
  ontology: JsonObject
) -> ConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze the following conflict between staging and production data.
    
    ## Conflict Details
    {{conflict}}
    
    ## Ontology Context
    {{ontology}}
    
    Provide a detailed analysis of this conflict, including:
    1. A clear description of the conflict
    2. Potential causes
    3. Semantic meaning
    4. Business impact
    5. Key differences
    
    Focus on providing actionable insights that would help in resolving this conflict.
  "#
}

function AnalyzeRelationshipPropertyConflict(
  relationship_type: string,
  property_name: string,
  staging_value: string,
  production_value: string,
  value_type: string,
  graph_context: string
) -> RelationshipConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this relationship property conflict between staging and production:

    Relationship Type: {{relationship_type}}
    Property Name: {{property_name}}
    Value Type: {{value_type}}
    
    Staging Value: {{staging_value}}
    Production Value: {{production_value}}
    
    Graph Context:
    {{graph_context}}
    
    Guidelines:
    1. Consider the property semantics within the relationship context
    2. Evaluate impact on relationship meaning and utility
    3. Assess risks to data integrity
    4. Consider query patterns that may be affected
    5. Evaluate if auto-resolution is safe
    
    {{ ctx.output_format }}
  "#
}

function AnalyzeMissingRelationshipConflict(
  relationship_type: string,
  source_entity_type: string,
  target_entity_type: string,
  source_entity_id: string,
  target_entity_id: string,
  exists_in: string,
  graph_context: string
) -> RelationshipConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this missing relationship conflict between staging and production:

    Relationship Type: {{relationship_type}}
    Source Entity Type: {{source_entity_type}}
    Target Entity Type: {{target_entity_type}}
    Source Entity ID: {{source_entity_id}}
    Target Entity ID: {{target_entity_id}}
    Exists in: {{exists_in}} (staging or production)
    
    Graph Context:
    {{graph_context}}
    
    Guidelines:
    1. Consider the importance of this relationship type
    2. Evaluate impact on graph connectivity
    3. Assess risks of adding or removing the relationship
    4. Consider query patterns that may be affected
    5. Evaluate if auto-resolution is safe
    
    {{ ctx.output_format }}
  "#
}

function AnalyzeRelationshipTypeConflict(
  staging_relationship_type: string,
  production_relationship_type: string,
  source_entity_type: string,
  target_entity_type: string,
  source_entity_id: string,
  target_entity_id: string,
  graph_context: string
) -> RelationshipConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this relationship type conflict between staging and production:

    Staging Relationship Type: {{staging_relationship_type}}
    Production Relationship Type: {{production_relationship_type}}
    Source Entity Type: {{source_entity_type}}
    Target Entity Type: {{target_entity_type}}
    Source Entity ID: {{source_entity_id}}
    Target Entity ID: {{target_entity_id}}
    
    Graph Context:
    {{graph_context}}
    
    Guidelines:
    1. Compare the semantic meaning of both relationship types
    2. Evaluate which type is more appropriate for these entities
    3. Assess risks of changing the relationship type
    4. Consider query patterns that may be affected
    5. Evaluate if auto-resolution is safe
    
    {{ ctx.output_format }}
  "#
}

function AnalyzeRelationshipDirectionConflict(
  relationship_type: string,
  staging_source_id: string,
  staging_target_id: string,
  production_source_id: string,
  production_target_id: string,
  graph_context: string
) -> RelationshipConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this relationship direction conflict between staging and production:

    Relationship Type: {{relationship_type}}
    Staging Source ID: {{staging_source_id}}
    Staging Target ID: {{staging_target_id}}
    Production Source ID: {{production_source_id}}
    Production Target ID: {{production_target_id}}
    
    Graph Context:
    {{graph_context}}
    
    Guidelines:
    1. Consider the semantic meaning of the relationship direction
    2. Evaluate which direction is more appropriate
    3. Assess risks of changing the relationship direction
    4. Consider query patterns that may be affected
    5. Evaluate if auto-resolution is safe
    
    {{ ctx.output_format }}
  "#
}

function AnalyzeDuplicateEntityConflict(
  entity_type: string,
  staging_entity_id: string,
  production_entity_ids: string,
  similarity_scores: string,
  entity_properties: string
) -> ConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this duplicate entity conflict where a staging entity matches multiple production entities:

    Entity Type: {{entity_type}}
    Staging Entity ID: {{staging_entity_id}}
    Production Entity IDs: {{production_entity_ids}}
    Similarity Scores: {{similarity_scores}}
    
    Entity Properties:
    {{entity_properties}}
    
    Guidelines:
    1. Evaluate whether these entities truly represent the same real-world object
    2. Consider the implications of merging vs. keeping separate
    3. Assess which production entity is the best match if any
    4. Identify potential data quality issues
    5. Recommend whether this requires manual review
    
    Provide a detailed analysis of this entity duplication conflict, including:
    1. A clear description of why multiple matches were found
    2. Potential causes for these duplicate entities
    3. The semantic meaning of these matches
    4. Business impact of having duplicate entities
    5. Key similarities and differences between the matching entities
    
    {{ ctx.output_format }}
  "#
}

function AnalyzeEntityMatchConflict(
  entity_type: string,
  staging_entity_id: string,
  production_entity_id: string,
  match_confidence: float,
  staging_properties: string,
  production_properties: string
) -> ConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this entity match conflict where a staging entity potentially matches a production entity:

    Entity Type: {{entity_type}}
    Staging Entity ID: {{staging_entity_id}}
    Production Entity ID: {{production_entity_id}}
    Match Confidence: {{match_confidence}}
    
    Staging Properties:
    {{staging_properties}}
    
    Production Properties:
    {{production_properties}}
    
    Guidelines:
    1. Evaluate whether these entities truly represent the same real-world object
    2. Consider the key identifying properties and their similarity
    3. Assess the implications of treating them as the same vs. different entities
    4. Identify potential data quality issues
    5. Consider the confidence score in your analysis
    
    Provide a detailed analysis of this entity match conflict, including:
    1. A clear description of why these entities might be matches
    2. Potential causes for having potentially duplicate entities
    3. The semantic meaning of this potential match
    4. Business impact of treating them as same or different
    5. Key similarities and differences between the entities
    
    {{ ctx.output_format }}
  "#
}

function AnalyzeSchemaConflict(
  entity_type: string,
  conflict_description: string,
  staging_schema: string,
  production_schema: string,
  affected_properties: string
) -> ConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this schema conflict between staging and production:

    Entity Type: {{entity_type}}
    Conflict Description: {{conflict_description}}
    
    Staging Schema:
    {{staging_schema}}
    
    Production Schema:
    {{production_schema}}
    
    Affected Properties:
    {{affected_properties}}
    
    Guidelines:
    1. Evaluate the nature of the schema differences
    2. Consider the impact on data integrity and consistency
    3. Assess the implications for queries and applications
    4. Identify potential migration challenges
    5. Consider backward compatibility issues
    
    Provide a detailed analysis of this schema conflict, including:
    1. A clear description of the schema differences
    2. Potential causes for the schema divergence
    3. The semantic meaning of these schema changes
    4. Business impact of the schema conflict
    5. Key considerations for resolution
    
    {{ ctx.output_format }}
  "#
}

function GenerateSchemaConflictResolutionOptions(
  conflict_analysis: string,
  entity_type: string,
  staging_schema: string,
  production_schema: string
) -> ResolutionOption[] {
  client DynamicGemini
  prompt #"
    For this schema conflict:

    Analysis:
    {{conflict_analysis}}

    Entity Type:
    {{entity_type}}

    Staging Schema:
    {{staging_schema}}

    Production Schema:
    {{production_schema}}

    Generate 3-5 distinct resolution options. For each option, provide:
    1. A short, descriptive name for the approach
    2. A detailed explanation of the schema migration steps
    3. The business impact of this resolution
    4. Any risks or considerations for data integrity
    5. Implementation complexity

    All options should be valid solutions with different trade-offs.
    Be specific and actionable in your suggestions.

    {{ ctx.output_format }}
  "#
}

function GenerateEntityMatchResolutionOptionsFromStagingAndProd(
  conflict_analysis: string,
  entity_type: string,
  staging_entity_id: string,
  production_entity_id: string,
  staging_properties: string,
  production_properties: string
) -> ResolutionOption[] {
  client DynamicGemini
  prompt #"
    For this entity match conflict:

    Analysis:
    {{conflict_analysis}}

    Entity Type:
    {{entity_type}}

    Staging Entity ID:
    {{staging_entity_id}}

    Production Entity ID:
    {{production_entity_id}}

    Staging Properties:
    {{staging_properties}}

    Production Properties:
    {{production_properties}}

    Generate 3-5 distinct resolution options. For each option, provide:
    1. A short, descriptive name for the approach
    2. A detailed explanation of the resolution steps
    3. The business impact of this resolution
    4. Any risks or considerations for data integrity
    5. Implementation complexity

    Consider options such as:
    - Treating them as the same entity (merge)
    - Keeping them as separate entities
    - Creating a relationship between them
    - Partial property merging

    All options should be valid solutions with different trade-offs.
    Be specific and actionable in your suggestions.

    {{ ctx.output_format }}
  "#
}

function AnalyzeEntityMatchWithOntology(
  conflict: JsonObject,
  ontology: JsonObject
) -> ConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze the following entity match conflict where a staging entity potentially matches a production entity.
    
    ## Conflict Details
    {{conflict}}
    
    ## Ontology Context
    {{ontology}}
    
    Provide a detailed analysis of this entity match conflict, including:
    1. A clear description of why these entities might be matches
    2. Potential causes for having potentially duplicate entities
    3. The semantic meaning of this potential match in the context of the ontology
    4. Business impact of treating them as same or different
    5. Key similarities and differences between the entities
    
    Focus on understanding whether these entities truly represent the same real-world object,
    considering the ontology constraints and business rules.
  "#
}

function AnalyzeSchemaConflictWithOntology(
  conflict: JsonObject,
  ontology: JsonObject
) -> ConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze the following schema conflict between staging and production.
    
    ## Conflict Details
    {{conflict}}
    
    ## Ontology Context
    {{ontology}}
    
    Provide a detailed analysis of this schema conflict, including:
    1. A clear description of the schema differences
    2. Potential causes for the schema divergence
    3. The semantic meaning of these schema changes in the context of the ontology
    4. Business impact of the schema conflict
    5. Key considerations for resolution
    
    Focus on understanding how this schema conflict affects data integrity, query capabilities,
    and compliance with the ontology constraints.
  "#
}

function AnalyzeEntityMatchResolutionOption(
  entity_type: string,
  resolution_option: string,
  staging_entity_id: string,
  production_entity_id: string,
  staging_properties: string,
  production_properties: string
) -> ConflictAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this specific resolution option for an entity match conflict:

    Entity Type: {{entity_type}}
    Resolution Option: {{resolution_option}}
    Staging Entity ID: {{staging_entity_id}}
    Production Entity ID: {{production_entity_id}}
    
    Staging Properties:
    {{staging_properties}}
    
    Production Properties:
    {{production_properties}}
    
    Guidelines:
    1. Evaluate the implications of this specific resolution approach
    2. Consider data quality and integrity impacts
    3. Assess business process impacts
    4. Identify potential risks and mitigations
    5. Consider implementation complexity
    
    Provide a detailed analysis of this resolution option, including:
    1. A clear description of what the resolution entails
    2. Potential benefits of this approach
    3. Potential drawbacks or risks
    4. Impact on data quality and integrity
    5. Recommendations for implementation
    
    {{ ctx.output_format }}
  "#
}

function BatchAnalyzeConflicts(
  conflict_type: string,
  entity_type: string,
  conflict_summaries: string,
  sample_conflicts: string,
  total_count: int
) -> ConflictGroupAnalysis {
  client DynamicGemini
  prompt #"
    Analyze this batch of similar conflicts to identify patterns and recommend a unified approach:

    Conflict Type: {{conflict_type}}
    Entity Type: {{entity_type}}
    Total Conflicts: {{total_count}}
    
    Conflict Summaries:
    {{conflict_summaries}}
    
    Sample Conflicts:
    {{sample_conflicts}}
    
    Guidelines:
    1. Identify common patterns across these conflicts
    2. Determine if they can be resolved with a single approach
    3. Evaluate risks of batch resolution
    4. Consider exceptions that might need special handling
    5. Assess confidence in a unified resolution strategy
    
    {{ ctx.output_format }}
  "#
}
