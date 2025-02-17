import httpx
import asyncio
from pathlib import Path
from datetime import datetime, timezone
import aiofiles
import json

from app.services.marker.models import (
    MarkerResponse,
    ConversionMetadata,
    HealthStatus
)

class MarkerAPIError(Exception):
    """Base exception for Marker API errors"""
    pass

class ConversionError(MarkerAPIError):
    """PDF conversion errors"""
    pass

class ValidationError(MarkerAPIError):
    """Markdown validation errors"""
    pass

class MarkerAPIClient:
    def __init__(
        self, 
        host: str,
        timeout: int = 120,
        max_retries: int = 3,
        backoff_factor: float = 0.5
    ):
        """
        Initialize Marker API client
        
        Args:
            host: Marker API host URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            backoff_factor: Exponential backoff factor
        """
        self.base_url = f"{host}/marker"
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
    def _init_session(self) -> httpx.AsyncClient:
        """Initialize HTTP session with retry configuration"""
        return httpx.AsyncClient(
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
    
    async def convert_to_markdown(
        self,
        file_path: Path,
        use_llm: bool = False,
        paginate: bool = True
    ) -> MarkerResponse:
        """
        Convert PDF file to markdown using Marker API
        
        Args:
            file_path: Path to PDF file
            use_llm: Whether to use LLM for conversion
            paginate: Whether to paginate output
            
        Returns:
            MarkerResponse with conversion results
            
        Raises:
            ConversionError: If conversion fails
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            async with self._init_session() as session:
                # Read file content
                async with aiofiles.open(file_path, 'rb') as f:
                    content = await f.read()
                
                # Prepare request
                files = {'file': (file_path.name, content, 'application/pdf')}
                data = {
                    'use_llm': json.dumps(use_llm),
                    'paginate': json.dumps(paginate)
                }
                
                # Make request with retries
                for attempt in range(self.max_retries):
                    try:
                        response = await session.post(
                            f"{self.base_url}/convert",
                            files=files,
                            data=data
                        )
                        response.raise_for_status()
                        break
                    except httpx.HTTPError as e:
                        if attempt == self.max_retries - 1:
                            raise ConversionError(f"Failed to convert PDF: {str(e)}")
                        await asyncio.sleep(self.backoff_factor * (2 ** attempt))
                
                # Process response
                result = response.json()
                conversion_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                return MarkerResponse(
                    status="success",
                    markdown_content=result["content"],
                    conversion_metadata=ConversionMetadata(
                        pages=len(result["content"]) if paginate else 1,
                        conversion_time=conversion_time,
                        file_size=len(content),
                        conversion_timestamp=datetime.now(timezone.utc),
                        settings={
                            "use_llm": use_llm,
                            "paginate": paginate
                        }
                    )
                )
                
        except Exception as e:
            raise ConversionError(f"Failed to convert PDF: {str(e)}")
    
    async def check_health(self) -> HealthStatus:
        """Check Marker API health status"""
        try:
            start_time = datetime.now(timezone.utc)
            async with self._init_session() as session:
                response = await session.get(f"{self.base_url}/health")
                response.raise_for_status()
                
                latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                return HealthStatus(
                    status="healthy",
                    latency=latency,
                    error_rate=0.0,
                    last_check=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                latency=0.0,
                error_rate=1.0,
                last_check=datetime.now(timezone.utc)
            )
