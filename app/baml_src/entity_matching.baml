class EntitySimilarityAnalysis {
    similarity_score float @description(#"
        Overall similarity score between entities (0-1).
        E.g: 0.95
    "#)
    
    confidence float @description(#"
        Confidence in the similarity assessment (0-1).
        E.g: 0.85
    "#)
    
    matching_properties string[] @description(#"
        List of properties that match between entities.
        E.g: ['name', 'email']
    "#)
    
    mismatched_properties string[] @description(#"
        List of properties that differ between entities.
        E.g: ['address', 'phone']
    "#)
    
    semantic_similarity float @description(#"
        Semantic similarity based on property values (0-1).
        E.g: 0.92
    "#)
    
    potential_merge_impact string @description(#"
        Analysis of impact if entities are merged.
        E.g: 'Low risk - entities appear to represent same real-world object'
    "#)
    
    recommended_action string @description(#"
        Recommended resolution action.
        E.g: 'merge_entities', 'keep_separate', 'needs_review'
    "#)
    
    reasoning string @description(#"
        Explanation of the analysis and recommendation.
        E.g: 'High property overlap and semantic similarity indicate same entity'
    "#)
}

function AnalyzeEntitySimilarity(
    entity_type: string,
    staging_properties: string,
    production_properties: string,
    domain_context: string
) -> EntitySimilarityAnalysis {
    client VertexAI
    prompt #"
        Analyze the similarity between these potentially matching entities:

        Entity Type: {{entity_type}}
        
        Staging Properties:
        {{staging_properties}}
        
        Production Properties:
        {{production_properties}}
        
        Domain Context:
        {{domain_context}}
        
        Guidelines:
        1. Compare property values semantically
        2. Consider domain-specific matching rules
        3. Evaluate merge impact and risks
        4. Assess confidence in similarity
        5. Provide clear reasoning
        
        {{ ctx.output_format }}
    "#
}

class MatchingNode {
    staging_node_id string @description(#"
        ID of the staging node.
        E.g: '12sds-2132d-21312f-fggfd3'
    "#)

    node_id string @description(#"
        ID of the matching node. Confidence must be > 0.9. If none match, return None
        E.g: '12sds-2132d-21312f-fggfd3'
    "#)
    
    confidence float @description(#"
        Confidence in the matching assessment (0-1).
        E.g: 0.85
    "#)
}

function GetMatchingNodes(
    candidate_sets: string[]
) -> MatchingNode[] {
    client VertexAI
    prompt #"
        Given multiple sets of staging node and production candidates, pick the most likely Production Node that matches each Staging Node within the set:

        Candidate Sets:
        {{candidate_sets}}

        Guidelines:
        1. Compare property values semantically and they should be about the same thing
        2. Consider domain-specific matching rules
        3. Evaluate merge impact and risks
        4. Assess confidence in similarity
        5. Pick the most likely Production Node that matches each Staging Node within the set

        {{ ctx.output_format }}
    "#
}
