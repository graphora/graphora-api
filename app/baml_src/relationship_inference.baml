// Defining a data model.
class RelationshipInference {
  id string @description(#"
    Should be of this format: <source_id>_<rel_type>_<target_id>
  "#)
  type string @description(#"
    The type of relationship.
  "#)
  source_id string @description(#"
    The ID of the source entity.
  "#)
  target_id string @description(#"
    The ID of the target entity.
  "#)
  source_type string @description(#"
    The type of the source entity.
  "#)
  target_type string @description(#"
    The type of the target entity.
  "#)
  properties map<string, string> @description(#"
    The properties of the relationship. Always pass an empty object.
  "#)
  confidence_score float @description(#"
    The confidence score of the relationship. A number between 0 and 1.
  "#)
}

// create function to infer relationships
function InferRelationship(rel_type: string,
  source_type: string,
  target_type: string,
  source_entities: string,
  target_entities: string,
  existing_rels: string
) -> RelationshipInference[] {
  client VertexAI
  prompt #"
    Given these potentially disconnected entities, identify any missing {{ rel_type }} relationships that would connect them:
                    
    Source entities ({{ source_type }}):
    {{ source_entities }}

    Target entities ({{target_type}}):
    {{ target_entities }}

    Existing relationships:
    {{ existing_rels }}

    Only return relationships that are clearly implied by the entity properties and context.
    Only return relationships if you are highly confident they exist based on the entity properties.

    {{ ctx.output_format}}
  "#
}