from pathlib import Path
import traceback
import aiofiles
from datetime import datetime
from prefect import task, get_run_logger
from typing import Optional

from app.services.marker.client import (
    MarkerAPIClient,
    ConversionError,
    MarkerResponse
)
from app.services.marker.models import ConversionResult
from app.config import settings

class ConversionStorage:
    def __init__(self, base_path: Path):
        """
        Initialize conversion storage
        
        Directory structure:
        base_path/
          └── {transform_id}/
              ├── original.pdf
              ├── markdown/
              │   └── markdown.md
              └── conversion_metadata.json
        """
        self.base_path = Path(base_path)
        
    async def store_conversion_result(
        self,
        transform_id: str,
        result: MarkerResponse,
        original_path: Path
    ) -> ConversionResult:
        """Store paginated markdown files"""
        transform_dir = self.base_path / transform_id
        markdown_dir = transform_dir / "markdown"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each page as separate markdown file
        md_path = markdown_dir / f"markdown.md"
        async with aiofiles.open(md_path, "w") as f:
            await f.write(result.markdown_content)
        
        # Save conversion metadata
        metadata_path = transform_dir / "conversion_metadata.json"
        async with aiofiles.open(metadata_path, "w") as f:
            await f.write(result.conversion_metadata.model_dump_json(indent=2))
        
        return ConversionResult(
            transform_id=transform_id,
            original_path=str(original_path),
            markdown_path=str(md_path),
            metadata=result.conversion_metadata,
            status="success"
        )

async def handle_conversion_error(
    error: Exception,
    transform_id: str,
    retry_count: int
) -> None:
    """Handle conversion errors"""
    logger = get_run_logger()
    
    error_context = {
        "transform_id": transform_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "retry_count": retry_count,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.error(
        f"Conversion error for transform {transform_id}",
        extra=error_context
    )
    
    # Store error context for recovery
    error_dir = Path(settings.UPLOAD_DIR) / transform_id / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    
    error_path = error_dir / f"error_{retry_count}.json"
    async with aiofiles.open(error_path, "w") as f:
        await f.write(json.dumps(error_context, indent=2))

@task(
    name="pdf-markdown-conversion",
    retries=3,
    retry_delay_seconds=30,
    tags=["marker-api"]
)
async def convert_pdf_to_markdown(
    file_path: Path,
    transform_id: str
) -> Optional[ConversionResult]:
    """
    Convert PDF to markdown using Marker API
    
    Args:
        file_path: Path to PDF file
        transform_id: Transform ID
        
    Returns:
        ConversionResult if successful, None if skipped
    """
    logger = get_run_logger()
    
    # Skip if not PDF
    if file_path.suffix.lower() != ".pdf":
        logger.info(f"Skipping non-PDF file: {file_path}")
        return None
    
    try:
        # Initialize client
        client = MarkerAPIClient(
            host=settings.MARKER_API_HOST,
            timeout=settings.MARKER_API_TIMEOUT,
            max_retries=settings.MARKER_API_MAX_RETRIES,
            backoff_factor=settings.MARKER_API_BACKOFF_FACTOR
        )
        logger.debug("Marker client configuration: %s", client)
        
        # Track conversion start
        logger.info(f"Starting PDF conversion for {transform_id}")
        
        # Convert to markdown
        result = await client.convert_to_markdown(
            file_path=file_path,
            use_llm=settings.MARKER_API_USE_LLM,
            paginate=settings.MARKER_API_PAGINATE
        )
        
        # Store result
        storage = ConversionStorage(Path(settings.UPLOAD_DIR))
        return await storage.store_conversion_result(
            transform_id=transform_id,
            result=result,
            original_path=file_path
        )
        
    except Exception as e:
        traceback.print_exc()
        await handle_conversion_error(e, transform_id, 0)
        raise ConversionError(f"Failed to convert PDF: {str(e)}")

@task(
    name="marker-api-health-check",
    retries=2,
    tags=["monitoring"]
)
async def check_marker_api_health() -> None:
    """Monitor Marker API health"""
    logger = get_run_logger()
    
    try:
        client = MarkerAPIClient(
            host=settings.MARKER_API_HOST,
            timeout=10  # Short timeout for health check
        )
        
        status = await client.check_health()
        logger.info(
            "Marker API health status",
            extra=status.model_dump()
        )
        
        # Alert if latency is too high
        if status.latency > 5000:  # 5 seconds
            logger.warning(
                f"High Marker API latency: {status.latency}ms",
                extra={"alert": "high_latency"}
            )
            
    except Exception as e:
        traceback.print_exc()
        logger.error(
            f"Marker API health check failed: {str(e)}",
            extra={"alert": "api_down"}
        )
