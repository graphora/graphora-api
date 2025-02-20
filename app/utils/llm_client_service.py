from typing import Any, Dict, Optional
from openai import OpenAI
import instructor
from pydantic import BaseModel
from app.config import settings
from openai import OpenAI
from tenacity import Retrying, stop_after_attempt, wait_fixed
import vertexai
from anthropic import AnthropicVertex
from vertexai.generative_models import GenerativeModel
from google.auth import default, transport
from datetime import datetime, timedelta
from mistralai_gcp import MistralGoogleCloud
from mistralai_gcp.utils.retries import RetryConfig, BackoffStrategy
import json
from google import genai
from typing import Type

class VertexAPIKey:
    def __init__(self):
        self.key = self.create_key()
        self.create_time = datetime.now()
    
    def create_key(self):
        credentials, _ = default()
        auth_request = transport.requests.Request()
        credentials.refresh(auth_request)
        return credentials.token
    
    def get_key(self):
        if datetime.now() - self.create_time > timedelta(minutes=50):
            self.key = self.create_key()
        return self.key

vertex_key_mgr = VertexAPIKey()

def extract(prompt: str, response_model: Type[BaseModel]) -> Type[BaseModel]:
    """Extract structured information using LLM"""
    client = genai.Client(
        vertexai=True, 
        project=settings.VERTEXAI_PROJECT_ID, 
        location=settings.VERTEXAI_LOCATION,
    )
    # Call model with structured output
    response = client.models.generate_content(
        model='gemini-2.0-flash-lite-preview-02-05',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': response_model,
            'temperature': 0,
        }
    )
    # Parse and validate response
    try:
        result = response.parsed
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {str(e)}")
    
def generate_text(prompt: str, json_response: bool = True) -> Optional[Dict]:
    """Generate text using LLM"""
    client = genai.Client(
        vertexai=True, 
        project=settings.VERTEXAI_PROJECT_ID, 
        location=settings.VERTEXAI_LOCATION,
    )
    # Call model with structured output
    if json_response:
        config = {
            'response_mime_type': 'application/json',
            'temperature': 0,
        }
    else:
        config = None
    response = client.models.generate_content(
        model='gemini-2.0-flash-lite-preview-02-05',
        contents=prompt,
        config=config,
    )
    # Parse and validate response
    try:
        result = json.loads(response.text) if json_response else response.text
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {str(e)}")

def call_llm(messages, response_model: Any):
        return call_llm_gemini(messages, response_model)
    
def call_llm_ollama(messages, response_model: Any, 
                        model = 'codestral:22b'):
    base_url = f"https://localhost:11434/v1"
    base_client = OpenAI(api_key="ollama", base_url=base_url)
    client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        temperature=0,
        max_tokens=8000,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        )
    )
    return response

def call_llm_codestral(messages, response_model: Any, 
                        model = 'codestral-2501'):
    PROJECT_ID=settings.VERTEXAI_PROJECT_ID
    LOCATION='us-central1'

    client = MistralGoogleCloud(access_token=vertex_key_mgr.get_key(), 
                                region=LOCATION, project_id=PROJECT_ID)
    resp = client.chat.complete(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
        timeout_ms=30000, 
        retries=RetryConfig(
            strategy="exponential", 
            backoff=BackoffStrategy(initial_interval=1,
                max_interval=2,
                exponent=2,
                max_elapsed_time=60), 
            retry_connection_errors=True)
    )
    return json.loads(resp.choices[0].message.content)
    
def call_llm_llama(messages, response_model: Any, 
                        model = 'meta/llama-3.3-70b-instruct-maas'):
    LOCATION='us-central1'
    base_url = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{settings.VERTEXAI_PROJECT_ID}/locations/{LOCATION}/endpoints/openapi"
    base_client = OpenAI(api_key=vertex_key_mgr.get_key(), base_url=base_url)
    client = instructor.from_openai(base_client, mode=instructor.Mode.VERTEXAI_TOOLS)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        temperature=0,
        max_tokens=8000,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        )
    )
    return response
    
def call_llm_deepseek(messages, response_model: Any, 
                        model = 'deepseek-chat', max_tokens=8000):
    base_client = OpenAI(api_key=settings.DEEPSEEK_API_KEY.strip(),
                                    base_url=settings.DEEPSEEK_BASE_URL.strip())
    client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        temperature=0,
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
        temperature=0,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        )
    )
    
def call_llm_gemini(messages, response_model, model = "gemini-2.0-flash-lite-preview-02-05"):
    PROJECT_ID=settings.VERTEXAI_PROJECT_ID
    LOCATION=settings.VERTEXAI_LOCATION

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    client = instructor.from_vertexai(
        client=GenerativeModel(model),
        mode=instructor.Mode.VERTEXAI_TOOLS,
    )
    return client.chat.completions.create(
        messages=messages,
        response_model=response_model,
        temperature=0,
        max_retries=Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(1),
        )
    )
    