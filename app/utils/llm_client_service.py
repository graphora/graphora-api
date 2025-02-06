from typing import Any
from openai import OpenAI
import instructor
from app.config import settings
from openai import OpenAI
from tenacity import Retrying, stop_after_attempt, wait_fixed
import vertexai
from anthropic import AnthropicVertex
from vertexai.generative_models import GenerativeModel

def call_llm(messages, response_model: Any):
        return call_llm_deepseek(messages, response_model)
    
def call_llm_deepseek(messages, response_model: Any, 
                        model = 'deepseek-chat', max_tokens=8000):
    base_client = OpenAI(api_key=settings.DEEPSEEK_API_KEY.strip(),
                                    base_url=settings.DEEPSEEK_BASE_URL.strip())
    client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        max_tokens=max_tokens,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        )
    )
    return response

def call_llm_anthropic(messages, response_model, 
                        model = 'claude-3-5-haiku@20241022',
                        max_tokens=8000):
    PROJECT_ID=settings.VERTEXAI_PROJECT_ID
    LOCATION='us-east5'

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
    llm = instructor.from_anthropic(client)
    return llm.messages.create(
        model=model,
        messages=messages,
        response_model=response_model,
        max_tokens=max_tokens,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        )
    )
    
def call_llm_gemini(messages, response_model, model = "gemini-2.0-flash-exp"):
    PROJECT_ID=settings.VERTEXAI_PROJECT_ID
    LOCATION=settings.VERTEXAI_LOCATION

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    client = instructor.from_vertexai(
        client=GenerativeModel(model),
        mode=instructor.Mode.VERTEXAI_TOOLS,
    )
    return client.chat.completions.create(
        messages=messages,
        response_model=response_model
    )
    
    