// Schema Generation BAML Functions
// This defines structured output for knowledge graph schema generation

// Entity Property Definition
class EntityProperty {
  type string
  description string
  required bool?
  unique bool?
  index bool?
}

// Relationship Property Definition  
class RelationshipProperty {
  type string
  description string
  required bool?
}

// Relationship Definition
class Relationship {
  target string
  properties map<string, RelationshipProperty>?
}

// Entity Definition
class Entity {
  properties map<string, EntityProperty>
  relationships map<string, Relationship>?
}

// Complete Schema Structure
class GeneratedSchema {
  version string
  entities map<string, Entity>
}

// Schema Generation Response
class SchemaGenerationResult {
  generated_schema GeneratedSchema
  confidence float
  suggestions string[]
  explanation string?
}

// Schema Generation Function
function GenerateKnowledgeGraphSchema(
  use_case: string,
  domain: string,
  data_sources: string,
  key_entities: string,
  relationships: string,
  query_patterns: string,
  data_complexity: string,
  data_volume: string,
  temporal_requirements: string,
  related_schemas: string
) -> SchemaGenerationResult {
  client DynamicGemini
  prompt #"
    You are an expert knowledge graph schema designer. Generate a structured schema based on the user's requirements.

    **User Requirements:**
    - Use Case: {{use_case}}
    - Domain: {{domain}}  
    - Data Sources: {{data_sources}}
    - Key Entities: {{key_entities}}
    - Relationships: {{relationships}}
    - Query Patterns: {{query_patterns}}
    - Data Complexity: {{data_complexity}}
    - Data Volume: {{data_volume}}
    - Temporal Requirements: {{temporal_requirements}}

    **Related Schemas for Reference:**
    {{related_schemas}}

    **CRITICAL OUTPUT FORMAT REQUIREMENTS:**
    You MUST return the response in the exact structured format expected:
    - generated_schema: A GeneratedSchema object with version "0.1.0" and entities map
    - confidence: A float between 0.0 and 1.0
    - suggestions: An array of strings with improvement suggestions
    - explanation: Optional string explaining the schema design

    **Schema Structure Requirements:**
    - The generated_schema must have a "version" field set to "0.1.0"
    - The generated_schema must have an "entities" field which is a map/dictionary
    - Each entity in the entities map must have:
      - properties: A map of property name to property definition
      - relationships: (optional) A map of relationship name to relationship definition
    - Each property must have: type, description, and optionally required, unique, index
    - Each relationship must have: target (entity name), and optionally properties map
    - Relationships MUST be nested inside their source entities, never at top level

    **Naming Conventions:**
    - Entity names: CamelCase (e.g., Customer, Product, Order)
    - Relationship names: UPPER_CASE (e.g., OWNS, HAS, RELATES_TO)
    - Property names: camelCase (e.g., firstName, createdAt, isActive)

    **Property Types:** Use only: str, int, float, bool, date

    **Design Guidelines:**
    1. Create entities based on the key entities mentioned or infer from use case
    2. Define properties for each entity based on the use case
    3. Create relationships that support the query patterns
    4. Add temporal fields if temporal requirements are specified
    5. Include audit fields (created_at, updated_at) for entities with high data volume
    6. Ensure relationships are properly nested inside source entities

    {{ ctx.output_format }}
  "#
}

// Schema Refinement Function
function RefineKnowledgeGraphSchema(
  current_schema_yaml: string,
  user_feedback: string,
  use_case: string,
  domain: string
) -> SchemaGenerationResult {
  client DynamicGemini
  prompt #"
    You are refining a knowledge graph schema based on user feedback.

    **Current Schema (YAML):**
    ```yaml
    {{current_schema_yaml}}
    ```

    **User Feedback:**
    {{user_feedback}}

    **Context:**
    - Use Case: {{use_case}}
    - Domain: {{domain}}

    **CRITICAL OUTPUT FORMAT REQUIREMENTS:**
    You MUST return the response in the exact structured format expected:
    - generated_schema: A GeneratedSchema object with version "0.1.0" and entities map
    - confidence: A float between 0.0 and 1.0
    - suggestions: An array of strings with improvement suggestions
    - explanation: String explaining the changes made

    **Schema Structure Requirements:**
    - The generated_schema must have a "version" field set to "0.1.0"
    - The generated_schema must have an "entities" field which is a map/dictionary
    - Each entity in the entities map must have:
      - properties: A map of property name to property definition
      - relationships: (optional) A map of relationship name to relationship definition
    - Each property must have: type, description, and optionally required, unique, index
    - Each relationship must have: target (entity name), and optionally properties map
    - Relationships MUST be nested inside their source entities, never at top level

    **Instructions:**
    1. Analyze the user's feedback carefully
    2. Make the requested changes while maintaining schema integrity
    3. Preserve existing structure unless explicitly asked to change
    4. Ensure all changes align with knowledge graph best practices
    5. Maintain proper entity-relationship nesting
    6. Follow naming conventions (CamelCase entities, UPPER_CASE relationships, camelCase properties)
    7. Property types must be: str, int, float, bool, date

    {{ ctx.output_format }}
  "#
}