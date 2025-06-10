class StandardisedProperties {
  entity_id string @description(#"
    The ID of the entity.
  "#)
  properties map<string, string> @description(#"
    The standardized properties of the entity.
    E.g:
    { 
      "property1": "standardized value",
      "property2": "standardized value"
    }
  "#)
}

function StandardiseProperties(entity_group_type: string, entities_json: string) -> StandardisedProperties[] {
  client DynamicGemini
  prompt #"
    You are processing a group of similar entities that should be standardized.
    Below are the properties of these entities:
    
    {{ entities_json }}
    
    Please standardize the properties by:
    1. Using the most accurate/complete VALUE (NOT PROPERTY NAME) for each property
    2. Fixing capitalization and formatting inconsistencies for values
    3. DO NOT CHANGE THE PROPERTY NAMES
    4. Include only the standardised properties & the corresponding entity IDs in the output

    {{ ctx.output_format }}
  "#
}