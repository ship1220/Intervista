# utils/logger.py
# Structured logging system

import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import traceback


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exc()
            }
        
        # Add extra fields if provided
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)
        
        return json.dumps(log_obj)


class Logger:
    """Structured logger wrapper."""
    
    def __init__(self, name: str = __name__, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        # Always filter at INFO level - suppress DEBUG logs
        self.logger.setLevel(logging.INFO)
        
        # Only add handlers if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
    
    def info(self, message: str, **extra):
        """Log info level."""
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.extra_fields = extra
        self.logger.handle(record)
    
    def debug(self, message: str, **extra):
        """Log debug level."""
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.extra_fields = extra
        self.logger.handle(record)
    
    def warning(self, message: str, **extra):
        """Log warning level."""
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.extra_fields = extra
        self.logger.handle(record)
    
    def error(self, message: str, exc_info=None, **extra):
        """Log error level."""
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=exc_info
        )
        record.extra_fields = extra
        self.logger.handle(record)
    
    def critical(self, message: str, **extra):
        """Log critical level."""
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.CRITICAL,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.extra_fields = extra
        self.logger.handle(record)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log structured event."""
        self.info(f"Event: {event_type}", event_type=event_type, **data)
    
    def log_prompt(self, prompt: str, max_chars: int = 200):
        """Log prompt (truncated)."""
        truncated = prompt[:max_chars] + "..." if len(prompt) > max_chars else prompt
        self.debug(f"Prompt: {truncated}", prompt_length=len(prompt))
    
    def log_response(self, response: str, max_chars: int = 200):
        """Log response (truncated)."""
        truncated = response[:max_chars] + "..." if len(response) > max_chars else response
        self.debug(f"Response: {truncated}", response_length=len(response))
