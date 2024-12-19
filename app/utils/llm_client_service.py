from typing import Optional
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import instructor
from app.config import settings
from app.utils.logger import logger

class LLMClientService:
    def __init__(self):
        self.client = None
        self.model = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize LLM client with instructor patch"""
        if not settings.OPENAI_API_KEY and not settings.ANTHROPIC_API_KEY and not settings.GOOGLE_GEMINI_API_KEY:
            logger.warning("LLM API key not set. LLM features will be unavailable.")
            return
        
        try:
            if settings.OPENAI_API_KEY.strip():
                base_client = OpenAI(api_key=settings.OPENAI_API_KEY.strip())
                self.client = instructor.from_openai(base_client)
                self.model = 'gpt-4'
            elif settings.ANTHROPIC_API_KEY.strip():
                base_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY.strip())
                self.client = instructor.from_anthropic(base_client)
                self.model = 'claude-3-5-haiku-20241022'
            else:
                genai.configure(api_key=settings.GOOGLE_GEMINI_API_KEY.strip())
                self.client = instructor.from_gemini(
                    client=genai.GenerativeModel(
                        model_name="models/gemini-1.5-flash-latest",
                    ),
                    mode=instructor.Mode.GEMINI_JSON,
                )
                self.model = 'models/gemini-1.5-flash-latest'
            logger.info(f"LLM client [{str(self.client.provider.name)}] initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            self.client = None

    @property
    def is_available(self) -> bool:
        """Check if LLM client is available"""
        return self.client is not None