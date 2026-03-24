from loguru import logger
from rich.logging import RichHandler
import os
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

SLIDERS_LOGS_DIR = os.environ.get("SLIDERS_LOGS_DIR")

# Remove the default loguru handler
logger.remove()

# Add a new handler using RichHandler for console output
logger.add(
    RichHandler(markup=True, show_time=False),  # Enable rich markup for colored output
    level="INFO",  # Set the logging level
    format="{message}",
    backtrace=True,  # Include the backtrace in the log
    diagnose=True,  # Include diagnostic information in the log
)

# Add another handler for saving debug logs in JSONL format
logger.add(
    os.path.join(
        SLIDERS_LOGS_DIR, "experiments/debug_logs_{time:YYYYMMDD_HHmm}.jsonl"
    ),  # Use .jsonl extension for JSON Lines format
    level="DEBUG",
    format='{{"time": "{time:YYYY-MM-DD HH:mm:ss}", "level": "{level.name}", "name": "{name}", "message": "{message}"}}',
    rotation="5 MB",
    retention=2,
    backtrace=True,
    diagnose=True,
    enqueue=True,
)
