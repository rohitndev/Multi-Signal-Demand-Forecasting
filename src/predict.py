"""
predict.py
----------
Load the saved TFT checkpoint and produce a 28-day P10/P50/P90 forecast for a
single store.

Run:
    python -m src.predict --store 1 --days 28
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CLEAN_CSV, MODEL_CKPT
from src.features import build_features

# Cache the loaded model so the API doesn't reload it on every request.
_MODEL = None


def _load_model():
    """Load (and memoise) the TFT model from the checkpoint."""
    global _MODEL
    if _MODEL is None:
        from pytorch_forecasting import TemporalFusionTransformer

        if not MODEL_CKPT.exists():
            raise FileNotFoundError(
                f"No model at {MODEL_CKPT}. Train first: python -m src.train"
            )
        _MODEL = TemporalFusionTransformer.load_from_checkpoint(str(MODEL_CKPT))
    return _MODEL


def _store_frame(store_id: int) -> pd.DataFrame:
    """Rebuild the engineered frame for a single store in the format TFT expects."""
    df = pd.read_csv(CLEAN_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Store"] == store_id].copy()
    if df.empty:
        raise ValueError(f"Store {store_id} not found in clean data.")

    df = build_features(df, run_tsfresh=False)
    df = df.sort_values("Date").reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))
    df["Sales"] = df["Sales"].astype("float32")  # match training dtype (softplus normalizer)
    df["Store"] = df["Store"].astype(str)
    df["month"] = df["month"].astype(str)
    df["DayOfWeek"] = df["DayOfWeek"].astype(str)
    return df


def forecast(store_id: int, forecast_days: int = 28) -> dict:
    """
    Return {store_id, forecasts: [{date, p10, p50, p90}, ...]} for `store_id`.
    The TFT predicts its trained horizon (28 days); we slice to forecast_days.
    """
    model = _load_model()
    df = _store_frame(store_id)

    # mode="quantiles" returns the configured [0.1, 0.5, 0.9] quantiles.
    raw = model.predict(df, mode="quantiles", return_x=False)
    q = np.asarray(raw)[0]  # shape: (horizon, 3)

    last_date = df["Date"].max()
    rows = []
    horizon = min(forecast_days, q.shape[0])
    for i in range(horizon):
        date = (last_date + pd.Timedelta(days=i + 1)).date().isoformat()
        rows.append(
            {
                "date": date,
                "p10": round(float(q[i, 0]), 2),
                "p50": round(float(q[i, 1]), 2),
                "p90": round(float(q[i, 2]), 2),
            }
        )

    return {"store_id": store_id, "forecasts": rows}


def _print_table(result: dict) -> None:
    print(f"\n28-day forecast for Store {result['store_id']}")
    print(f"{'Date':<12}{'P10':>10}{'P50':>10}{'P90':>10}")
    print("-" * 42)
    for r in result["forecasts"]:
        print(f"{r['date']:<12}{r['p10']:>10}{r['p50']:>10}{r['p90']:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast demand for one store.")
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--days", type=int, default=28)
    args = parser.parse_args()

    result = forecast(args.store, args.days)
    _print_table(result)
