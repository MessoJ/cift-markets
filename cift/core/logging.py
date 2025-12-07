"""
CIFT Markets - Logging Configuration

Structured logging using loguru with JSON formatting for production.
"""

import sys
from pathlib import Path

from loguru import logger

from cift.core.config import settings


def setup_logging() -> None:
    """Configure application-wide logging."""
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

    # File handler with rotation
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if settings.log_format == "json":
        logger.add(
            str(log_path),
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            level=settings.log_level,
            serialize=True,  # JSON format
            enqueue=True,  # Async logging
        )
    else:
        logger.add(
            str(log_path),
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            level=settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            enqueue=True,
        )

    logger.info(f"Logging configured: level={settings.log_level}, format={settings.log_format}")


# Initialize logging on module import
setup_logging()
