"""
run_all.py
----------
One-shot driver that runs the entire DS-02 pipeline end to end:

    ingest -> train (TFT) -> predict -> replenishment agent -> drift report

Usage (from the project root):
    python run_all.py                      # full run with defaults
    python run_all.py --synthetic          # generate synthetic data first
    python run_all.py --epochs 10 --store 3
    python run_all.py --skip-train         # reuse an existing checkpoint

Flags:
    --synthetic        Generate synthetic Rossmann-style CSVs before ingesting.
    --epochs N         TFT training epochs            (default 5)
    --n-stores N       Train on the first N stores    (default 6; 0 = all stores)
    --store ID         Store to forecast / replenish  (default 1)
    --inventory N      Current inventory for the agent (default 5000)
    --reorder-point N  Reorder threshold for the agent (default 1000)
    --skip-train       Skip training and use the saved checkpoint.
"""
from __future__ import annotations

import argparse
import sys
import time

from src.config import CLEAN_CSV, MODEL_CKPT, DATA_RAW


def _banner(step: str, title: str) -> None:
    """Print a clear section header so each stage is easy to spot in the log."""
    print("\n" + "=" * 70)
    print(f"  [{step}] {title}")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full DS-02 pipeline.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic Rossmann-style data first.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-stores", type=int, default=6,
                        help="Train on first N stores (0 = all).")
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--inventory", type=int, default=5000)
    parser.add_argument("--reorder-point", type=int, default=1000)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    started = time.time()

    # --- 0) Optional synthetic data -------------------------------------- #
    if args.synthetic:
        _banner("0/5", "Generating synthetic dataset")
        from scripts.make_synthetic_data import main as make_data
        make_data()

    # Make sure raw data exists before we start.
    if not (DATA_RAW / "train.csv").exists():
        print(
            f"\nERROR: no dataset found in {DATA_RAW}.\n"
            "Either pass --synthetic to generate test data, or download the\n"
            "Rossmann CSVs (train.csv + store.csv) into data/raw/. See README."
        )
        return 1

    # --- 1) Ingestion ---------------------------------------------------- #
    _banner("1/5", "Ingestion (load, merge, clean, validate)")
    from src.ingestion import run as ingest
    ingest()

    # --- 2) Train -------------------------------------------------------- #
    n_stores = None if args.n_stores == 0 else args.n_stores
    if args.skip_train:
        _banner("2/5", "Training (SKIPPED — reusing existing checkpoint)")
        if not MODEL_CKPT.exists():
            print(f"ERROR: --skip-train set but no checkpoint at {MODEL_CKPT}.")
            return 1
    else:
        _banner("2/5", f"Training TFT ({args.epochs} epochs, "
                        f"{'all' if n_stores is None else n_stores} stores)")
        from src.train import train
        mape = train(epochs=args.epochs, n_stores=n_stores)
        print(f"\n>>> Training MAPE: {mape:.2f}%")

    # --- 3) Predict ------------------------------------------------------ #
    _banner("3/5", f"Forecasting store {args.store} (28-day P10/P50/P90)")
    from src.predict import forecast, _print_table
    result = forecast(args.store, forecast_days=28)
    _print_table(result)

    # --- 4) Replenishment agent ----------------------------------------- #
    _banner("4/5", f"Replenishment agent (inventory={args.inventory})")
    from src.agent import replenish
    p50 = [row["p50"] for row in result["forecasts"]]
    replenish(
        store_id=args.store,
        p50_forecast=p50,
        current_inventory=args.inventory,
        reorder_point=args.reorder_point,
    )

    # --- 5) Drift report ------------------------------------------------- #
    _banner("5/5", "Drift detection (reference vs current window)")
    from src.drift import run as drift_run
    drift_run()

    # --- Done ------------------------------------------------------------ #
    elapsed = time.time() - started
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Clean data : {CLEAN_CSV}")
    print(f"  Checkpoint : {MODEL_CKPT}")
    print("  Next: serve it with  ->  uvicorn src.api:app --reload")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
