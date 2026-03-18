"""COVID-19 Prediction Models Module.

Implements multiple ML models for COVID-19 case forecasting:
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor
- ARIMA time series model
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CovidModelTrainer:
    """Trains and evaluates ML models for COVID-19 prediction."""

    FEATURE_EXCLUDE = ["date", "state", "confirmed", "deaths"]
    TARGET_COL = "new_confirmed"

    MODEL_REGISTRY = {
        "linear": LinearRegression,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
    }

    MODEL_PARAMS = {
        "linear": {},
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1,
        },
        "gradient_boosting": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        },
    }

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        model_type: str = "all",
        forecast_days: int = 14,
    ) -> Dict[str, Dict[str, Any]]:
        """Train models and return evaluation metrics."""
        X, y = self._prepare_data(df)
        logger.info(f"Training data shape: {X.shape}")

        models_to_train = (
            list(self.MODEL_REGISTRY.keys())
            if model_type == "all"
            else [model_type]
        )

        results = {}
        for name in models_to_train:
            logger.info(f"Training model: {name}")
            metrics = self._train_single_model(X, y, name)
            results[name] = metrics
            logger.info(
                f"{name} - RMSE: {metrics['rmse']:.2f}, "
                f"MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.4f}"
            )

        return results

    def _prepare_data(
        self, df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable."""
        feature_cols = [
            col for col in df.columns
            if col not in self.FEATURE_EXCLUDE
            and df[col].dtype in ["float64", "int64", "int32"]
        ]
        X = df[feature_cols].drop(columns=[self.TARGET_COL], errors="ignore")
        y = df[self.TARGET_COL]
        return X, y

    def _train_single_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
    ) -> Dict[str, Any]:
        """Train a single model with time series cross-validation."""
        model_class = self.MODEL_REGISTRY[model_name]
        params = self.MODEL_PARAMS[model_name]

        scaler = StandardScaler()
        tscv = TimeSeriesSplit(n_splits=5)

        fold_metrics: List[Dict[str, float]] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train = scaler.fit_transform(X.iloc[train_idx])
            X_val = scaler.transform(X.iloc[val_idx])
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = model_class(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            predictions = np.maximum(predictions, 0)

            fold_metrics.append({
                "rmse": np.sqrt(mean_squared_error(y_val, predictions)),
                "mae": mean_absolute_error(y_val, predictions),
                "r2": r2_score(y_val, predictions),
            })

        avg_metrics = {
            key: np.mean([m[key] for m in fold_metrics])
            for key in fold_metrics[0]
        }
        avg_metrics["model_name"] = model_name
        avg_metrics["n_features"] = X.shape[1]
        avg_metrics["n_samples"] = len(y)

        return avg_metrics
