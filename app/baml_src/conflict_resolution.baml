// Models for conflict resolution

// Output types
class ResolutionOption {
  id string @description(#"
    Unique identifier for this resolution option
  "#)
  description string @description(#"
    Human-readable description of the resolution option
    E.g: 'Keep the staging value (35) as it appears to be more recent'
  "#)
  resolution_type string @description(#"
    Type of resolution strategy.
    Must be one of: KEEP_STAGING, KEEP_PRODUCTION, MERGE_VALUES, CREATE_NEW, CUSTOM
  "#)
  resolution_data JsonObject @description(#"
    Data needed to implement this resolution
  "#)
  confidence float @description(#"
    Confidence score for this resolution option (0.0 to 1.0)
  "#)
  reasoning string @description(#"
    Explanation of why this option is recommended
  "#)
}

class ResolutionOptions {
  options ResolutionOption[] @description(#"
    List of possible resolution options
  "#)
  recommended_option_id string @description(#"
    ID of the recommended option
  "#)
  recommendation_reasoning string @description(#"
    Explanation of why the recommended option is preferred
  "#)
}

class SelectedResolution {
  option_id string @description(#"
    ID of the selected resolution option
  "#)
  resolution_type string @description(#"
    Type of resolution strategy
  "#)
  resolution_data JsonObject @description(#"
    Data needed to implement this resolution
  "#)
  confidence float @description(#"
    Confidence in this resolution (0.0 to 1.0)
  "#)
}

function GeneratePropertyResolutionOptions(
  conflict: JsonObject,
  analysis: JsonObject,
  ontology: JsonObject
) -> ResolutionOptions {
  client DynamicGemini
  prompt #"
    Generate resolution options for the following property conflict.
    
    ## Conflict Details
    {{conflict}}
    
    ## Conflict Analysis
    {{analysis}}
    
    ## Ontology Context
    {{ontology}}
    
    Based on the conflict details and analysis, generate a comprehensive set of resolution options.
    For each option, provide:
    1. A unique ID
    2. A clear description
    3. The resolution type (KEEP_STAGING, KEEP_PRODUCTION, MERGE_VALUES, CREATE_NEW, or CUSTOM)
    4. The necessary data to implement the resolution
    5. A confidence score (0.0 to 1.0)
    6. Reasoning for this option
    
    Then, recommend the best option and explain your reasoning.
    
    Focus on providing practical, implementable resolution strategies that maintain data integrity.
  "#
}

function GenerateRelationshipResolutionOptions(
  conflict: JsonObject,
  analysis: JsonObject,
  ontology: JsonObject
) -> ResolutionOptions {
  client DynamicGemini
  prompt #"
    Generate resolution options for the following relationship conflict.
    
    ## Conflict Details
    {{conflict}}
    
    ## Conflict Analysis
    {{analysis}}
    
    ## Ontology Context
    {{ontology}}
    
    Based on the conflict details and analysis, generate a comprehensive set of resolution options.
    For each option, provide:
    1. A unique ID
    2. A clear description
    3. The resolution type (KEEP_STAGING, KEEP_PRODUCTION, MERGE_VALUES, CREATE_NEW, or CUSTOM)
    4. The necessary data to implement the resolution
    5. A confidence score (0.0 to 1.0)
    6. Reasoning for this option
    
    Then, recommend the best option and explain your reasoning.
    
    Focus on maintaining relationship integrity and ensuring that the graph remains consistent.
  "#
}

function GenerateEntityMatchResolutionOptions(
  conflict: JsonObject,
  analysis: JsonObject,
  ontology: JsonObject
) -> ResolutionOptions {
  client DynamicGemini
  prompt #"
    Generate resolution options for the following entity matching conflict.
    
    ## Conflict Details
    {{conflict}}
    
    ## Conflict Analysis
    {{analysis}}
    
    ## Ontology Context
    {{ontology}}
    
    Based on the conflict details and analysis, generate a comprehensive set of resolution options.
    For each option, provide:
    1. A unique ID
    2. A clear description
    3. The resolution type (KEEP_STAGING, KEEP_PRODUCTION, MERGE_VALUES, CREATE_NEW, or CUSTOM)
    4. The necessary data to implement the resolution
    5. A confidence score (0.0 to 1.0)
    6. Reasoning for this option
    
    Then, recommend the best option and explain your reasoning.
    
    Consider options like merging the entities, keeping one of the production entities, or creating a new entity.
    Focus on maintaining data integrity and reducing duplication.
  "#
}

function GenerateGenericResolutionOptions(
  conflict: JsonObject,
  analysis: JsonObject,
  ontology: JsonObject
) -> ResolutionOptions {
  client DynamicGemini
  prompt #"
    Generate resolution options for the following conflict.
    
    ## Conflict Details
    {{conflict}}
    
    ## Conflict Analysis
    {{analysis}}
    
    ## Ontology Context
    {{ontology}}
    
    Based on the conflict details and analysis, generate a comprehensive set of resolution options.
    For each option, provide:
    1. A unique ID
    2. A clear description
    3. The resolution type (KEEP_STAGING, KEEP_PRODUCTION, MERGE_VALUES, CREATE_NEW, or CUSTOM)
    4. The necessary data to implement the resolution
    5. A confidence score (0.0 to 1.0)
    6. Reasoning for this option
    
    Then, recommend the best option and explain your reasoning.
    
    Focus on providing practical, implementable resolution strategies that maintain data integrity.
  "#
}

function SelectBestResolution(
  conflict: JsonObject,
  options: JsonObject,
  ontology: JsonObject
) -> SelectedResolution {
  client DynamicGemini
  prompt #"
    Select the best resolution option for the following conflict.
    
    ## Conflict Details
    {{conflict}}
    
    ## Resolution Options
    {{options}}
    
    ## Ontology Context
    {{ontology}}
    
    Based on the conflict details and available options, select the best resolution option.
    Provide:
    1. The ID of the selected option
    2. The resolution type
    3. The resolution data needed for implementation
    4. Your confidence in this selection (0.0 to 1.0)
    
    Focus on selecting the option that best maintains data integrity, follows business rules,
    and resolves the conflict in the most appropriate way.
  "#
}
