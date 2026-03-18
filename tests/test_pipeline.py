"""Unit tests for COVID-19 prediction pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-03-15", periods=60, freq="D")
    confirmed = np.cumsum(np.random.randint(50, 500, size=60))
    deaths = (confirmed * np.random.uniform(0.02, 0.05, size=60)).astype(int)
    new_confirmed = np.diff(confirmed, prepend=0)
    new_deaths = np.diff(deaths, prepend=0)

    return pd.DataFrame({
        "date": dates,
        "state": "SP",
        "confirmed": confirmed,
        "deaths": deaths,
        "new_confirmed": new_confirmed,
        "new_deaths": new_deaths,
    })


class TestCovidDataIngestion:
    """Tests for data ingestion module."""

    def test_load_sample_data(self):
        """Test loading sample CSV data."""
        from src.data_ingestion import CovidDataIngestion
        ingestion = CovidDataIngestion()
        df = ingestion.load_sample_data()
        assert not df.empty
        assert "date" in df.columns

    def test_validate_missing_columns(self, sample_df):
        """Test validation raises error for missing columns."""
        from src.data_ingestion import CovidDataIngestion
        ingestion = CovidDataIngestion()
        bad_df = sample_df.drop(columns=["confirmed"])
        with pytest.raises(ValueError, match="Missing required columns"):
            ingestion._validate_dataframe(bad_df)


class TestCovidFeatureEngineer:
    """Tests for feature engineering module."""

    def test_create_features(self, sample_df):
        """Test full feature engineering pipeline."""
        from src.feature_engineering import CovidFeatureEngineer
        engineer = CovidFeatureEngineer()
        result = engineer.create_features(sample_df)
        assert len(result.columns) > len(sample_df.columns)
        assert "day_of_week" in result.columns

    def test_temporal_features(self, sample_df):
        """Test temporal feature extraction."""
        from src.feature_engineering import CovidFeatureEngineer
        engineer = CovidFeatureEngineer()
        result = engineer._add_temporal_features(sample_df.copy())
        assert "is_weekend" in result.columns
        assert result["is_weekend"].isin([0, 1]).all()


class TestCovidModelTrainer:
    """Tests for ML model training module."""

    def test_prepare_data(self, sample_df):
        """Test data preparation for modeling."""
        from src.feature_engineering import CovidFeatureEngineer
        from src.models import CovidModelTrainer
        engineer = CovidFeatureEngineer()
        df = engineer.create_features(sample_df)
        trainer = CovidModelTrainer()
        X, y = trainer._prepare_data(df)
        assert len(X) == len(y)
        assert "date" not in X.columns

    def test_train_linear_model(self, sample_df):
        """Test training linear regression model."""
        from src.feature_engineering import CovidFeatureEngineer
        from src.models import CovidModelTrainer
        engineer = CovidFeatureEngineer()
        df = engineer.create_features(sample_df)
        trainer = CovidModelTrainer()
        results = trainer.train_and_evaluate(df, model_type="linear")
        assert "linear" in results
        assert "rmse" in results["linear"]
        assert results["linear"]["rmse"] >= 0
