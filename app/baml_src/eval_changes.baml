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
        Only do corrections if you are more than 95% confident that the change is incorrect.
        Corrections should be relevant and not out of context.
    "#)
    confidence_score float @description(#"
        Overall confidence score for all the corrections (0-1)
        E.g: 0.95
    "#)
}

function EvalChanges(
    change_logs: string,
    past_resolutions: string
) -> ChangeResult[] {
    client DynamicGemini
    prompt #"
        You are a helpful assistant that evaluates the changes made to a graph.
        You are given a list of change logs and learnings from past resolutions.
        You need to evaluate the changes and determine if they are correct.
        Please validate the changes against the past resolutions and make corrections if necessary.
        Please return the ID of the change log and the corrections needed.

        Past Resolutions & Learnings:
        {{past_resolutions}}

        Change Logs:
        {{change_logs}}

        {{ ctx.output_format }}
    "#
}
