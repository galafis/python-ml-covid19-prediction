"""COVID-19 Data Ingestion Module.

Handles data loading from CSV files and optional API fetching
from public COVID-19 data sources (brasil.io).
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


class CovidDataIngestion:
    """Manages COVID-19 data acquisition and validation."""

    API_URL = "https://brasil.io/api/dataset/covid19/caso_full/data/"
    REQUIRED_COLUMNS = [
        "date", "state", "confirmed", "deaths",
        "new_confirmed", "new_deaths",
    ]

    def load_sample_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load COVID-19 data from local CSV file."""
        if filepath is None:
            filepath = DATA_DIR / "sample_covid_data.csv"

        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, parse_dates=["date"])
        self._validate_dataframe(df)
        logger.info(f"Loaded {len(df)} records successfully")
        return df

    def fetch_from_api(
        self, state: str = "SP", timeout: int = 30,
    ) -> pd.DataFrame:
        """Fetch COVID-19 data from brasil.io API."""
        logger.info(f"Fetching data for state: {state}")
        params = {"state": state, "is_last": "False"}
        headers = {"User-Agent": "covid-ml-prediction/1.0"}

        try:
            response = requests.get(
                self.API_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json().get("results", [])

            if not data:
                logger.warning("API returned empty results, using sample data")
                return self.load_sample_data()

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            self._validate_dataframe(df)
            logger.info(f"Fetched {len(df)} records from API")
            return df

        except (requests.RequestException, ValueError) as exc:
            logger.error(f"API request failed: {exc}")
            logger.info("Falling back to sample data")
            return self.load_sample_data()

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame contains required columns."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df.empty:
            raise ValueError("DataFrame is empty")

        null_counts = df[self.REQUIRED_COLUMNS].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values detected:\n{null_counts[null_counts > 0]}")
