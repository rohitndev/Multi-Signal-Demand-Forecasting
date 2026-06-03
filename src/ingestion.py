"""
ingestion.py
------------
Load the raw Rossmann CSVs, merge store metadata, clean, validate, and
write a single tidy CSV that the rest of the pipeline consumes.

Run:
    python -m src.ingestion
"""
import sys

import pandas as pd

from src.config import DATA_RAW, CLEAN_CSV


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train.csv and store.csv from data/raw/."""
    train_path = DATA_RAW / "train.csv"
    store_path = DATA_RAW / "store.csv"

    if not train_path.exists() or not store_path.exists():
        sys.exit(
            f"Missing dataset. Put train.csv and store.csv in {DATA_RAW}\n"
            "See README for the Kaggle download steps."
        )

    # low_memory=False avoids dtype guessing warnings on the mixed columns.
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path, low_memory=False)
    return train, store


def clean(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Merge, parse dates, drop closed days, and validate key columns."""
    # 1) Merge store attributes onto each daily sales row.
    df = train.merge(store, on="Store", how="left")

    # 2) Parse the Date column to real datetimes and sort chronologically.
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # 3) Drop days the store was closed (Sales are 0 and add no signal).
    df = df[df["Open"] == 1].copy()

    # 4) Validate: the columns we depend on must not contain nulls.
    required = ["Sales", "Store", "Date"]
    nulls = df[required].isnull().sum()
    if nulls.any():
        raise ValueError(f"Null values found in required columns:\n{nulls}")

    # CompetitionDistance has a few genuine NaNs; fill so downstream code is safe.
    if "CompetitionDistance" in df.columns:
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
            df["CompetitionDistance"].median()
        )

    return df


def run() -> pd.DataFrame:
    """Full ingestion pipeline: load -> clean -> save -> report."""
    train, store = load_raw()
    df = clean(train, store)

    df.to_csv(CLEAN_CSV, index=False)

    print("Ingestion complete.")
    print(f"  Rows (open days)   : {len(df):,}")
    print(f"  Stores             : {df['Store'].nunique()}")
    print(f"  Date range         : {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"  Saved clean CSV to : {CLEAN_CSV}")
    return df


if __name__ == "__main__":
    run()
