import aiohttp
import aiofiles
import asyncio
from pathlib import Path
from datetime import datetime, timezone
import logging
import json
from typing import Optional

from app.services.marker.models import (
    MarkerResponse,
    ConversionMetadata,
    HealthStatus
)

logger = logging.getLogger(__name__)

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
        self.base_url = host
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
    def _init_session(self) -> aiohttp.ClientSession:
        """Initialize HTTP session with timeout"""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        return aiohttp.ClientSession(timeout=timeout)

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
                async with aiofiles.open(file_path, "rb") as f:
                    content = await f.read()
                
                # Prepare multipart form data
                data = aiohttp.FormData()
                data.add_field('file',
                            content,
                            filename=file_path.name,
                            content_type='application/pdf')
                data.add_field('use_llm', 'False')
                data.add_field('paginate_output', 'True')
                data.add_field('force_ocr', 'False')
                
                # Make request with retries
                for attempt in range(self.max_retries):
                    try:
                        print(f"Starting PDF conversion attempt {attempt + 1}/{self.max_retries}")
                        logger.info(f"Starting PDF conversion attempt {attempt + 1}/{self.max_retries}")
                        async with session.post(f"{self.base_url}/marker/upload", data=data) as response:
                            if response.status >= 400:
                                error_text = await response.text()
                                raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                            
                            result = await response.json()
                            logger.info("PDF conversion completed successfully")
                            break
                    except aiohttp.ClientError as e:
                        if attempt == self.max_retries - 1:
                            raise ConversionError(f"Failed to convert PDF: {str(e)}")
                        logger.warning(f"PDF conversion attempt {attempt + 1} failed: {str(e)}")
                        await asyncio.sleep(self.backoff_factor * (2 ** attempt))
                
                # Process response
                conversion_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                return MarkerResponse(
                    status="success",
                    markdown_content=result["output"],
                    conversion_metadata=ConversionMetadata(
                        pages=len(result["metadata"]["page_stats"]) if paginate else 1,
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
            async with self._init_session() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        return HealthStatus(status="healthy")
                    else:
                        return HealthStatus(status="unhealthy")
        except Exception as e:
            return HealthStatus(status="unhealthy", error=str(e))
