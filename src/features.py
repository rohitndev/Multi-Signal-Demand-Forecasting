"""
features.py
-----------
Turn the clean merged DataFrame into a model-ready feature matrix.

Engineered features:
  * Lags:        sales_lag_7, sales_lag_14, sales_lag_28
  * Rolling:     sales_rolling_mean_7, sales_rolling_std_7, sales_rolling_mean_28
  * Calendar:    day_of_week, month, is_weekend
  * Promo:       promo_flag, days_since_last_promo
  * tsfresh:     a small set of automatic per-store features (MinimalFCParameters)

Run:
    python -m src.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLEAN_CSV


def _add_lags_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Per-store lag and rolling-window features on Sales."""
    g = df.groupby("Store")["Sales"]

    df["sales_lag_7"] = g.shift(7)
    df["sales_lag_14"] = g.shift(14)
    df["sales_lag_28"] = g.shift(28)

    # Rolling stats are computed on the *shifted* series to avoid leaking the
    # current day's sales into its own features.
    shifted = g.shift(1)
    df["sales_rolling_mean_7"] = shifted.rolling(7).mean().reset_index(0, drop=True)
    df["sales_rolling_std_7"] = shifted.rolling(7).std().reset_index(0, drop=True)
    df["sales_rolling_mean_28"] = shifted.rolling(28).mean().reset_index(0, drop=True)
    return df


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week, month and weekend flag from the Date column."""
    df["Date"] = pd.to_datetime(df["Date"])
    # DayOfWeek already exists in Rossmann (1=Mon..7=Sun); keep a 0-based copy too.
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def _add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Promo flag plus 'days since the last promo' per store."""
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    df["promo_flag"] = df["Promo"].astype(int)

    # Vectorised "days since last promo": take the Date on promo days, forward-fill
    # it within each store, then measure the gap. Robust for 1 or many stores
    # (groupby.apply returning a Series vs DataFrame is version-dependent).
    promo_dates = df["Date"].where(df["promo_flag"] == 1)
    last_promo = promo_dates.groupby(df["Store"]).ffill()
    df["days_since_last_promo"] = (
        (df["Date"] - last_promo).dt.days.fillna(0).astype(int)
    )
    return df


def _add_tsfresh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run tsfresh with MinimalFCParameters on the Sales series per store and
    merge the resulting static features back onto every row of that store.
    Kept minimal so it stays fast enough for a college laptop.
    """
    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters

        # tsfresh wants a long frame: id (Store), time index, value.
        ts = df[["Store", "Date", "Sales"]].copy()
        ts = ts.sort_values(["Store", "Date"])
        ts["time_idx"] = ts.groupby("Store").cumcount()

        extracted = extract_features(
            ts[["Store", "time_idx", "Sales"]],
            column_id="Store",
            column_sort="time_idx",
            column_value="Sales",
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True,
            n_jobs=0,  # 0 = single process; avoids multiprocessing issues on Windows.
        )
        extracted = extracted.add_prefix("tsf_").reset_index()
        extracted = extracted.rename(columns={"index": "Store"})
        # Clean up infinities / NaNs tsfresh sometimes produces.
        extracted = extracted.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df = df.merge(extracted, on="Store", how="left")
    except Exception as exc:  # noqa: BLE001 - keep pipeline running without tsfresh
        print(f"[features] tsfresh skipped ({exc}).")
    return df


def build_features(df: pd.DataFrame, run_tsfresh: bool = True) -> pd.DataFrame:
    """Apply all feature steps and return the final matrix (NaNs filled with 0)."""
    df = df.copy()
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    df = _add_lags_and_rolling(df)
    df = _add_calendar(df)
    df = _add_promo_features(df)
    if run_tsfresh:
        df = _add_tsfresh(df)

    # Lag/rolling features create NaNs for the first rows of each store.
    feature_cols = [
        "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "sales_rolling_mean_7", "sales_rolling_std_7", "sales_rolling_mean_28",
    ]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    return df


def run() -> pd.DataFrame:
    """Load clean.csv, build features, and report the shape."""
    df = pd.read_csv(CLEAN_CSV)
    feats = build_features(df)
    print("Feature engineering complete.")
    print(f"  Rows    : {len(feats):,}")
    print(f"  Columns : {feats.shape[1]}")
    print(f"  New cols: {[c for c in feats.columns if c.startswith(('sales_', 'tsf_', 'promo_', 'days_'))][:10]} ...")
    return feats


if __name__ == "__main__":
    run()
