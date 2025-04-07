"""
Logging utilities for the XAIR system.
"""

import os
import sys
import logging
import time
from typing import Optional, Union, Dict, Any
from pathlib import Path
from datetime import datetime
import json

# Try to import rich for enhanced console logging
try:
    from rich.console import Console
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class XAIRLogger:
    """Enhanced logging setup for XAIR system."""
    
    DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __init__(
        self,
        name: str = "xair",
        log_level: Union[int, str] = logging.INFO,
        log_file: Optional[str] = None,
        console_output: bool = True,
        use_rich: bool = True,
        capture_warnings: bool = True,
        log_format: Optional[str] = None
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_level: Logging level (e.g., logging.INFO)
            log_file: Path to log file
            console_output: Whether to output to console
            use_rich: Whether to use rich formatting for console output
            capture_warnings: Whether to capture Python warnings
            log_format: Custom log format string
        """
        self.name = name
        self.log_level = self._parse_log_level(log_level)
        self.log_file = log_file
        self.console_output = console_output
        self.use_rich = use_rich and RICH_AVAILABLE
        self.capture_warnings = capture_warnings
        self.log_format = log_format or self.DEFAULT_LOG_FORMAT
        
        # Create the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False
        
        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Add handlers
        if console_output:
            self._add_console_handler()
        
        if log_file:
            self._add_file_handler()
        
        # Capture warnings if requested
        if capture_warnings:
            logging.captureWarnings(True)
            
        # Log initial message
        self.logger.info(f"Logger initialized: {name}")
    
    def _parse_log_level(self, level: Union[int, str]) -> int:
        """Convert string log level to numeric value."""
        if isinstance(level, int):
            return level
        
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        
        return level_map.get(level.lower(), logging.INFO)
    
    def _add_console_handler(self) -> None:
        """Add console handler to logger."""
        if self.use_rich:
            # Use rich for pretty console output
            console = Console()
            handler = RichHandler(
                console=console,
                show_time=False,
                show_path=False,
                markup=True
            )
        else:
            # Standard console handler
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(self.log_format)
            handler.setFormatter(formatter)
        
        handler.setLevel(self.log_level)
        self.logger.addHandler(handler)
    
    def _add_file_handler(self) -> None:
        """Add file handler to logger."""
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(self.log_format)
        handler.setFormatter(formatter)
        handler.setLevel(self.log_level)
        self.logger.addHandler(handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger


class TimingLogger:
    """Utility for logging execution times of code blocks."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
        """
        Initialize the timing logger.
        
        Args:
            logger: Logger to use
            level: Logging level for timing messages
        """
        self.logger = logger or logging.getLogger("timing")
        self.level = level
        self.start_times = {}
        self.timings = {}
    
    def start(self, name: str) -> None:
        """
        Start timing a code block.
        
        Args:
            name: Name for the timing block
        """
        self.start_times[name] = time.time()
    
    def stop(self, name: str) -> float:
        """
        Stop timing a code block and log the elapsed time.
        
        Args:
            name: Name of the timing block
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.start_times:
            self.logger.warning(f"Timer '{name}' was never started")
            return 0.0
        
        elapsed = time.time() - self.start_times[name]
        self.timings[name] = elapsed
        
        self.logger.log(self.level, f"⏱️ {name}: {elapsed:.3f}s")
        return elapsed
    
    def get_timing(self, name: str) -> float:
        """
        Get the timing for a specific block.
        
        Args:
            name: Name of the timing block
            
        Returns:
            Elapsed time in seconds
        """
        return self.timings.get(name, 0.0)
    
    def get_all_timings(self) -> Dict[str, float]:
        """
        Get all recorded timings.
        
        Returns:
            Dictionary of all timings
        """
        return self.timings.copy()
    
    def reset(self) -> None:
        """Reset all timings."""
        self.start_times.clear()
        self.timings.clear()
    
    def summary(self) -> str:
        """
        Create a summary of all timings.
        
        Returns:
            Summary string
        """
        if not self.timings:
            return "No timings recorded."
        
        # Sort by elapsed time (descending)
        sorted_timings = sorted(
            self.timings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate total time
        total_time = sum(self.timings.values())
        
        # Build summary
        lines = ["Timing Summary:"]
        lines.append(f"Total time: {total_time:.3f}s")
        lines.append("-" * 40)
        
        for name, elapsed in sorted_timings:
            percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
            lines.append(f"{name}: {elapsed:.3f}s ({percentage:.1f}%)")
        
        return "\n".join(lines)


class LogStats:
    """Utility for collecting and logging statistics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the stats logger.
        
        Args:
            logger: Logger to use
        """
        self.logger = logger or logging.getLogger("stats")
        self.stats = {}
        self.start_time = time.time()
    
    def add(self, key: str, value: Any) -> None:
        """
        Add a statistic value.
        
        Args:
            key: Statistic name
            value: Value to record
        """
        self.stats[key] = value
    
    def increment(self, key: str, amount: Union[int, float] = 1) -> None:
        """
        Increment a counter statistic.
        
        Args:
            key: Counter name
            amount: Amount to increment by
        """
        if key not in self.stats:
            self.stats[key] = 0
        
        self.stats[key] += amount
    
    def average(self, key: str, value: Union[int, float]) -> None:
        """
        Add a value to calculate an average.
        
        Args:
            key: Average name
            value: Value to add
        """
        if key not in self.stats:
            self.stats[key] = {"sum": 0, "count": 0}
        
        self.stats[key]["sum"] += value
        self.stats[key]["count"] += 1
    
    def max(self, key: str, value: Union[int, float]) -> None:
        """
        Update maximum value.
        
        Args:
            key: Maximum name
            value: Value to compare
        """
        if key not in self.stats:
            self.stats[key] = value
        else:
            self.stats[key] = max(self.stats[key], value)
    
    def min(self, key: str, value: Union[int, float]) -> None:
        """
        Update minimum value.
        
        Args:
            key: Minimum name
            value: Value to compare
        """
        if key not in self.stats:
            self.stats[key] = value
        else:
            self.stats[key] = min(self.stats[key], value)
    
    def get(self, key: str) -> Any:
        """
        Get a statistic value.
        
        Args:
            key: Statistic name
            
        Returns:
            Statistic value
        """
        if key not in self.stats:
            return None
        
        if isinstance(self.stats[key], dict) and "sum" in self.stats[key] and "count" in self.stats[key]:
            # Calculate average
            if self.stats[key]["count"] > 0:
                return self.stats[key]["sum"] / self.stats[key]["count"]
            return 0
        
        return self.stats[key]
    
    def log(self, level: int = logging.INFO) -> None:
        """
        Log all statistics.
        
        Args:
            level: Logging level
        """
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Process averages
        stats_to_log = {}
        for key, value in self.stats.items():
            if isinstance(value, dict) and "sum" in value and "count" in value:
                # Calculate average
                if value["count"] > 0:
                    stats_to_log[key] = value["sum"] / value["count"]
                else:
                    stats_to_log[key] = 0
            else:
                stats_to_log[key] = value
        
        # Add elapsed time
        stats_to_log["elapsed_time"] = f"{elapsed:.3f}s"
        
        # Log stats
        self.logger.log(level, f"Stats: {json.dumps(stats_to_log, indent=2)}")
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.stats.clear()
        self.start_time = time.time()


def setup_logger(
    name: str = "xair",
    level: Union[int, str] = logging.INFO,
    log_dir: Optional[str] = None,
    use_rich: bool = True
) -> logging.Logger:
    """
    Set up a logger with standard configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        use_rich: Whether to use rich formatting
        
    Returns:
        Configured logger
    """
    log_file = None
    if log_dir:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Create and configure logger
    logger_config = XAIRLogger(
        name=name,
        log_level=level,
        log_file=log_file,
        console_output=True,
        use_rich=use_rich,
        capture_warnings=True
    )
    
    return logger_config.get_logger()