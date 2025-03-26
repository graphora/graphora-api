// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

client<llm> VertexAI {
  provider vertex-ai
  options {
    model env.VERTEXAI_DEFAULT_MODEL
    location env.VERTEXAI_LOCATION
    project_id env.VERTEXAI_PROJECT_ID

    generationConfig {
      temperature 0
    }
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/retry
retry_policy Constant {
  max_retries 3
  // Strategy is optional
  strategy {
    type constant_delay
    delay_ms 200
  }
}

retry_policy Exponential {
  max_retries 2
  // Strategy is optional
  strategy {
    type exponential_backoff
    delay_ms 300
    mutliplier 1.5
    max_delay_ms 10000
  }
}