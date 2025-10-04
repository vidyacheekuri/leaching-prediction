"""
21-Elements Monolithic Cement Leaching Prediction

A machine learning project for predicting elemental leaching behavior 
from monolithic cement materials under different pH and time conditions.
"""

__version__ = "1.0.0"
__author__ = "Research Assistant"
__email__ = "your-email@domain.com"

from .data_processing import DataProcessor
from .ml_pipeline import MLPipeline

__all__ = [
    "DataProcessor",
    "MLPipeline",
]
