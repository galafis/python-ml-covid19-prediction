"""COVID-19 ML Prediction - Source package."""

from src.data_ingestion import CovidDataIngestion
from src.feature_engineering import CovidFeatureEngineer
from src.models import CovidModelTrainer
from src.pipeline import CovidPredictionPipeline

__all__ = [
    "CovidDataIngestion",
    "CovidFeatureEngineer",
    "CovidModelTrainer",
    "CovidPredictionPipeline",
]

__version__ = "1.0.0"
