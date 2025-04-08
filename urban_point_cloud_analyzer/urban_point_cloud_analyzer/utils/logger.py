# urban_point_cloud_analyzer/utils/logger.py
import logging
import os
from pathlib import Path
from typing import Union

def setup_logger(log_file: Union[str, Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger.
    
    Args:
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger('urban_point_cloud_analyzer')
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger