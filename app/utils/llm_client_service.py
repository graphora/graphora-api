from typing import Optional
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import instructor
from app.config import settings
from app.utils.logger import logger
import vertexai
import google.auth
import google.auth.transport.requests
from vertexai.generative_models import GenerativeModel

class LLMClientService:
    def __init__(self):
        self.client = None
        self.model = None
        self.provider = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize LLM client with instructor patch"""

        if not settings.OPENAI_API_KEY and not settings.ANTHROPIC_API_KEY and not settings.GOOGLE_GEMINI_API_KEY and not settings.VERTEXAI_PROJECT_ID:
            logger.warning("LLM API key not set. LLM features will be unavailable.")
            return
        
        try:
            if settings.OPENAI_API_KEY.strip():
                base_client = OpenAI(api_key=settings.OPENAI_API_KEY.strip())
                self.client = instructor.from_openai(base_client)
                self.model = 'gpt-4'
                self.provider = 'openai'
            elif settings.ANTHROPIC_API_KEY.strip():
                base_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY.strip())
                self.client = instructor.from_anthropic(base_client)
                self.model = 'claude-3-5-haiku-20241022'
                self.provider = 'anthropic'
            elif settings.GOOGLE_GEMINI_API_KEY.strip():
                genai.configure(api_key=settings.GOOGLE_GEMINI_API_KEY.strip())
                self.client = instructor.from_gemini(
                    client=genai.GenerativeModel(
                        model_name="models/gemini-1.5-flash-latest",
                    ),
                    mode=instructor.Mode.GEMINI_JSON,
                )
                self.model = 'models/gemini-1.5-flash-latest'
                self.provider = 'gemini'
            else:
                creds, project = google.auth.default()
                auth_req = google.auth.transport.requests.Request()
                creds.refresh(auth_req)
                PROJECT=settings.VERTEXAI_PROJECT_ID
                LOCATION=settings.VERTEXAI_LOCATION
                base_url = f'https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT}/locations/{LOCATION}/endpoints/openapi'
                self.client = instructor.from_openai(
                    OpenAI(base_url=base_url, api_key=creds.token), mode=instructor.Mode.JSON
                )

                # vertexai.init(project=settings.VERTEXAI_PROJECT_ID)
                # self.client = instructor.from_vertexai(
                #     client=GenerativeModel("claude-3-5-sonnet-v2@20241022"),
                #     mode=instructor.Mode.VERTEXAI_JSON,
                # )
                self.model = 'google/gemini-2.0-flash-exp'
                self.provider = 'vertexai'
            logger.info(f"LLM client [{str(self.client.provider.name)}] initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            self.client = None

    @property
    def is_available(self) -> bool:
        """Check if LLM client is available"""
        return self.client is not None