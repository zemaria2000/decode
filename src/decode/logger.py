"""
    Imported by every file, it sets up logging for the entire project
"""

import logging
import os
from datetime import datetime

# ==========================================================
# LOGGING SETUP
# ==========================================================

# --- 1. Locate project root (folder above xai_tree) ---
PROJECT_ROOT = os.getcwd()

# --- 2. Define logs directory at project root ---
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# --- 3. Create timestamped log file ---
timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
log_file = os.path.join(LOG_DIR, f"tree_build_{timestamp}.log")

# --- 4. Configure logger ---
logger = logging.getLogger("xai_tree")
logger.setLevel(logging.DEBUG)
logger.propagate = False  

if not logger.handlers:  # Prevent duplicate logs when re-imported
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding = "utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger.info(f"Logging initialized — writing to {log_file}")