import logging
import json
import os
from datetime import datetime
from src.config import Config

class JSONFormatter(logging.Formatter):
    """Formats log records into structured JSON strings for log management tools."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        # Include exception tracebacks if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

def get_logger(name: str) -> logging.Logger:
    """Configures and returns a dual-handler structured logger."""
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger
        
    # Set base level from configuration
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level = log_level_map.get(Config.LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    
    # 1. Console Handler (Human-readable, colored/clean layout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 2. File Handler (JSON structured layout for audit logs)
    log_file_path = Config.LOG_FILE
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
    try:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file handler for {log_file_path}: {str(e)}")
        
    return logger

if __name__ == "__main__":
    # Test logger output
    logger = get_logger("TestLogger")
    logger.info("Structured logging pipeline initialized successfully.")
    logger.warning("Simulating transient sensor signal warning.")
    try:
        raise ValueError("Simulated mechanical overvoltage fault.")
    except Exception:
        logger.exception("An exception occurred during operations:")
