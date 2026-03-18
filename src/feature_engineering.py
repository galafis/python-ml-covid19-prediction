"""COVID-19 Feature Engineering Module.

Creates temporal, epidemiological, and statistical features
for predictive modeling of COVID-19 case trajectories.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CovidFeatureEngineer:
    """Generates predictive features from COVID-19 time series data."""

    ROLLING_WINDOWS = [7, 14, 21]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full feature engineering pipeline."""
        logger.info("Starting feature engineering")
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        df = self._add_temporal_features(df)
        df = self._add_rolling_statistics(df)
        df = self._add_growth_rates(df)
        df = self._add_epidemiological_features(df)
        df = self._add_lag_features(df)

        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values")
        logger.info(f"Final feature count: {len(df.columns)}")
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract calendar-based features from date column."""
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
        return df

    def _add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling mean and standard deviation."""
        for window in self.ROLLING_WINDOWS:
            for col in ["new_confirmed", "new_deaths"]:
                df[f"{col}_ma{window}"] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                df[f"{col}_std{window}"] = (
                    df[col].rolling(window=window, min_periods=1).std()
                )
        return df

    def _add_growth_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily and weekly growth rates."""
        for col in ["confirmed", "deaths"]:
            prev = df[col].shift(1)
            df[f"{col}_daily_growth"] = np.where(
                prev > 0, (df[col] - prev) / prev, 0.0,
            )
            prev_week = df[col].shift(7)
            df[f"{col}_weekly_growth"] = np.where(
                prev_week > 0, (df[col] - prev_week) / prev_week, 0.0,
            )
        return df

    def _add_epidemiological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add case fatality rate and doubling time estimates."""
        df["case_fatality_rate"] = np.where(
            df["confirmed"] > 0, df["deaths"] / df["confirmed"], 0.0,
        )
        growth = df["confirmed_daily_growth"].replace(0, np.nan)
        df["doubling_time"] = np.log(2) / np.log(1 + growth)
        df["doubling_time"] = df["doubling_time"].clip(0, 365).fillna(365)
        return df

    def _add_lag_features(
        self, df: pd.DataFrame, lags: List[int] = None,
    ) -> pd.DataFrame:
        """Create lagged versions of key columns."""
        if lags is None:
            lags = [1, 3, 7, 14]
        for lag in lags:
            for col in ["new_confirmed", "new_deaths"]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        return df
