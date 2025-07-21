class Correction {
    prop_name string @description(#"
        Name of the property that is being changed.
        E.g: 'name'
    "#)
    prop_value string @description(#"
        Value of the property that is being changed.
        E.g: 'John Doe'
    "#)
}

class ChangeResult {
    id string @description(#"
        ID of the change log.
        E.g: '123'
    "#)
    corrections Correction[] @description(#"
        Any corrections to the change log. If no corrections are needed, return an empty list.
        Corrections should be logical or based on past resolutions and learnings. If there is change in value for e.g price=100 to price=200, then it needs a manual review.
        In this case, the confidence should be 50% or less. But if its a simple typo or a minor change, then it can be corrected with high confidence.
        Only do corrections if you are more than 95% confident that the change is incorrect.
        Corrections should be relevant and not out of context.
    "#)
    confidence_score float @description(#"
        Overall confidence score for all the corrections (0-1). 
        E.g: 0.95
    "#)
}

function EvalChanges(
    change_logs: string,
    past_resolutions: string
) -> ChangeResult[] {
    client DynamicGemini
    prompt #"
        You are evaluating property changes in a knowledge graph merge.

          IMPORTANT GUIDELINES:
          - Only make corrections for obvious errors (typos, formatting issues)
          - For value changes (e.g., price: 100 -> 200), set confidence ≤ 0.5 to trigger human review
          - Do NOT auto-correct factual differences - let humans decide
          - Only correct if >95% confident it's an ERROR, not a legitimate update

          Examples of CORRECTIONS (high confidence):
          - "Jhon" -> "John" (typo)
          - "2023-13-45" -> "2023-12-31" (invalid date)

          Examples of HUMAN REVIEW (low confidence ≤ 0.5):
          - Medical codes: "847.2" -> "12000" (could be legitimate update)
          - Prices: "$100" -> "$200" (business decision)
          - Status changes: "Active" -> "Inactive" (operational change)

          Past Resolutions & Learnings:
          {{past_resolutions}}

          Change Logs:
          {{change_logs}}

        {{ ctx.output_format }}
    "#
}
