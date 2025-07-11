"""Telecoms Customer Churn ML Pipeline

A comprehensive machine learning pipeline for predicting customer churn
in the telecommunications industry.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data import DataProcessor
from .models import ChurnPredictor

__all__ = ["DataProcessor", "ChurnPredictor"]
