import logging
from typing import Union

from app.config import settings


def _coerce_log_level(level: Union[str, int]) -> int:
    """Convert string log levels to the numeric values expected by logging."""

    if isinstance(level, int):
        return level

    if isinstance(level, str):
        normalized = level.upper()
        if normalized.isdigit():
            return int(normalized)
        return getattr(logging, normalized, logging.INFO)

    return logging.INFO


# Configure logging
logging.basicConfig(
    level=_coerce_log_level(settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
