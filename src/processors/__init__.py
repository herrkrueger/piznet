"""
Processors Package - Clean Architecture
Data processing layer with standardized interfaces
"""

from .base import BaseProcessor, ProcessorResult, ProcessorPipeline

__all__ = ['BaseProcessor', 'ProcessorResult', 'ProcessorPipeline']