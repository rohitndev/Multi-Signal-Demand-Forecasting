"""
drift.py
--------
Use Evidently AI to detect feature drift between a reference window (first 70%
of the data) and a current window (last 30%). Also flags whether MAPE on the
current window degraded by more than 3 percentage points vs the reference.

Run:
    python -m src.drift
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLEAN_CSV
from src.features import build_features

# Columns we monitor for drift (numeric demand-related features).
MONITORED = [
    "Sales", "Customers", "Promo",
    "sales_lag_7", "sales_rolling_mean_7", "sales_rolling_mean_28",
    "day_of_week", "month",
]


def _naive_mape(window: pd.DataFrame) -> float:
    """
    Cheap proxy 'model error' for a window: predict each day with the 7-day
    lag and measure MAPE. Lets us compare reference vs current without loading
    the TFT. Lower is better.
    """
    actual = window["Sales"].to_numpy(float)
    pred = window["sales_lag_7"].to_numpy(float)
    mask = (actual != 0) & (pred != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def run() -> dict:
    """Compute drift report + MAPE degradation flag. Returns a JSON-able dict."""
    df = pd.read_csv(CLEAN_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = build_features(df, run_tsfresh=False)

    # 70/30 chronological split.
    split = int(len(df) * 0.7)
    reference = df.iloc[:split].copy()
    current = df.iloc[split:].copy()

    cols = [c for c in MONITORED if c in df.columns]
    ref_feat = reference[cols].fillna(0.0)
    cur_feat = current[cols].fillna(0.0)

    drifted_features: list[str] = []
    drift_score = 0.0

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_feat, current_data=cur_feat)
        result = report.as_dict()

        # DataDriftPreset emits several metrics: one carries the dataset-level
        # share of drifted columns, another carries the per-column table. Scan
        # all of them so we don't depend on their order.
        by_col: dict = {}
        for metric in result["metrics"]:
            res = metric.get("result", {})
            if "share_of_drifted_columns" in res:
                drift_score = float(res["share_of_drifted_columns"])
            if "drift_by_columns" in res:
                by_col = res["drift_by_columns"]

        drifted_features = [
            name for name, info in by_col.items() if info.get("drift_detected")
        ]
    except Exception as exc:  # noqa: BLE001 - fall back to a simple mean-shift test
        print(f"[drift] Evidently unavailable ({exc}) -> using mean-shift fallback.")
        for c in cols:
            ref_mean, cur_mean = ref_feat[c].mean(), cur_feat[c].mean()
            denom = abs(ref_mean) + 1e-9
            if abs(cur_mean - ref_mean) / denom > 0.25:  # >25% mean shift
                drifted_features.append(c)
        drift_score = len(drifted_features) / max(1, len(cols))

    # MAPE degradation flag.
    ref_mape = _naive_mape(reference)
    cur_mape = _naive_mape(current)
    mape_flag = bool(
        not np.isnan(ref_mape)
        and not np.isnan(cur_mape)
        and (cur_mape - ref_mape) > 3.0
    )

    out = {
        "drifted_features": drifted_features,
        "drift_score": round(drift_score, 4),
        "reference_mape": round(ref_mape, 2),
        "current_mape": round(cur_mape, 2),
        "mape_flag": mape_flag,
    }

    print("\n--- Drift Report ---")
    print(f"Drifted features : {drifted_features}")
    print(f"Drift score      : {out['drift_score']}")
    print(f"Reference MAPE   : {out['reference_mape']}%")
    print(f"Current MAPE     : {out['current_mape']}%")
    print(f"MAPE degraded >3%: {mape_flag}")
    return out


if __name__ == "__main__":
    run()
