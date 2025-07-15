"""
GP-based portfolio optimization package.
"""
from .config import GPModelConfig
from .run_pipeline import run_full_pipeline

__version__ = "1.0.0"
__all__ = ["GPModelConfig", "run_full_pipeline"]
