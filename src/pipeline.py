"""COVID-19 Prediction Pipeline - End-to-end orchestration."""

import logging
from typing import Dict, Any

import pandas as pd
import numpy as np
from pathlib import Path

from src.data_ingestion import CovidDataIngestion
from src.feature_engineering import CovidFeatureEngineer
from src.models import CovidModelTrainer

logger = logging.getLogger(__name__)


class CovidPredictionPipeline:
    """End-to-end COVID-19 prediction pipeline."""

    def __init__(
        self,
        state: str = "SP",
        fetch_data: bool = False,
        model_type: str = "all",
        forecast_days: int = 14,
        output_dir: str = "output",
    ) -> None:
        self.state = state.upper()
        self.fetch_data = fetch_data
        self.model_type = model_type
        self.forecast_days = forecast_days
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ingestion = CovidDataIngestion()
        self.engineer = CovidFeatureEngineer()
        self.trainer = CovidModelTrainer()

    def run(self) -> Dict[str, Dict[str, Any]]:
        """Execute the full prediction pipeline."""
        logger.info(f"Running pipeline for state: {self.state}")

        # Step 1: Data Ingestion
        logger.info("Step 1/4: Data Ingestion")
        if self.fetch_data:
            df = self.ingestion.fetch_from_api(state=self.state)
        else:
            df = self.ingestion.load_sample_data()
        logger.info(f"Loaded {len(df)} records")

        # Step 2: Feature Engineering
        logger.info("Step 2/4: Feature Engineering")
        df = self.engineer.create_features(df)
        logger.info(f"Created {len(df.columns)} features")

        # Step 3: Model Training
        logger.info("Step 3/4: Model Training")
        results = self.trainer.train_and_evaluate(
            df=df, model_type=self.model_type,
            forecast_days=self.forecast_days,
        )

        # Step 4: Save Results
        logger.info("Step 4/4: Saving Results")
        self._save_results(df, results)
        return results

    def _save_results(
        self, df: pd.DataFrame, results: Dict[str, Dict[str, Any]],
    ) -> None:
        """Save pipeline results to output directory."""
        output_path = self.output_dir / f"processed_{self.state}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

        metrics_df = pd.DataFrame(results).T
        metrics_path = self.output_dir / f"metrics_{self.state}.csv"
        metrics_df.to_csv(metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")
