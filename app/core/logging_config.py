"""
app/core/logging_config.py

Configures logging for the whole app. Call setup_logging() once at startup.
Every log line then looks like:
    2025-06-01 14:23:11 | INFO     | app.api.routes | Inference complete in 0.34s
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # These libraries are noisy at INFO — only show their warnings and errors
    for noisy in ("uvicorn.access", "ultralytics", "mlflow", "boto3", "botocore", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)