"""COVID-19 ML Prediction Pipeline - CLI Entry Point.

This module provides the command-line interface for running
the COVID-19 prediction pipeline with configurable options.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import CovidPredictionPipeline


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log"),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="COVID-19 ML Prediction Pipeline for Brazilian States",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        default=False,
        help="Fetch real data from OpenDataSUS API (default: use sample data)",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="SP",
        help="Brazilian state code to analyze (default: SP)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "xgboost", "prophet", "all"],
        default="all",
        help="Model to train (default: all)",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=14,
        help="Number of days to forecast (default: 14)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results (default: output)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the COVID-19 prediction pipeline."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting COVID-19 ML Prediction Pipeline")
    logger.info(f"State: {args.state} | Model: {args.model} | Forecast: {args.forecast_days}d")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipeline = CovidPredictionPipeline(
            state=args.state,
            fetch_data=args.fetch_data,
            model_type=args.model,
            forecast_days=args.forecast_days,
            output_dir=str(output_dir),
        )
        results = pipeline.run()

        logger.info("Pipeline completed successfully")
        for model_name, metrics in results.items():
            logger.info(
                f"{model_name}: MAPE={metrics.get('mape', 'N/A'):.2f}%, "
                f"RMSE={metrics.get('rmse', 'N/A'):.2f}, "
                f"R2={metrics.get('r2', 'N/A'):.4f}"
            )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
