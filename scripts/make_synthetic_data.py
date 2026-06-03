"""
make_synthetic_data.py
----------------------
Generate small synthetic train.csv / store.csv files in the SAME schema as the
Kaggle Rossmann dataset, so you can smoke-test the whole pipeline offline before
downloading the real data.

Run:
    python -m scripts.make_synthetic_data
"""
import numpy as np
import pandas as pd

from src.config import DATA_RAW

N_STORES = 6
N_DAYS = 220          # enough history for lag_28 / rolling_28 + a 28-day horizon
START = pd.Timestamp("2014-01-01")
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, periods=N_DAYS, freq="D")

    train_rows = []
    store_rows = []

    for store in range(1, N_STORES + 1):
        base = rng.integers(4000, 9000)          # per-store baseline demand
        store_rows.append(
            {
                "Store": store,
                "StoreType": rng.choice(list("abcd")),
                "Assortment": rng.choice(list("abc")),
                "CompetitionDistance": float(rng.integers(20, 20000)),
                "CompetitionOpenSinceMonth": int(rng.integers(1, 13)),
                "CompetitionOpenSinceYear": int(rng.integers(2000, 2014)),
                "Promo2": int(rng.integers(0, 2)),
                "Promo2SinceWeek": int(rng.integers(1, 52)),
                "Promo2SinceYear": int(rng.integers(2009, 2014)),
                "PromoInterval": rng.choice(["", "Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov"]),
            }
        )

        for d in dates:
            dow = d.dayofweek + 1                 # Rossmann uses 1=Mon..7=Sun
            open_flag = 0 if dow == 7 else 1      # closed Sundays
            promo = int(rng.random() < 0.4) if open_flag else 0

            # Demand = baseline * weekly seasonality * promo lift * noise.
            weekly = 1.0 + 0.15 * np.sin(2 * np.pi * dow / 7)
            promo_lift = 1.25 if promo else 1.0
            sales = int(base * weekly * promo_lift * rng.normal(1.0, 0.08)) if open_flag else 0
            sales = max(sales, 0)
            customers = int(sales / rng.uniform(8, 12)) if open_flag else 0

            train_rows.append(
                {
                    "Store": store,
                    "DayOfWeek": dow,
                    "Date": d.strftime("%Y-%m-%d"),
                    "Sales": sales,
                    "Customers": customers,
                    "Open": open_flag,
                    "Promo": promo,
                    "StateHoliday": "0",
                    "SchoolHoliday": int(rng.random() < 0.1),
                }
            )

    train_df = pd.DataFrame(train_rows)
    store_df = pd.DataFrame(store_rows)

    train_df.to_csv(DATA_RAW / "train.csv", index=False)
    store_df.to_csv(DATA_RAW / "store.csv", index=False)

    print("Synthetic Rossmann-style data written:")
    print(f"  {DATA_RAW / 'train.csv'}  ({len(train_df):,} rows)")
    print(f"  {DATA_RAW / 'store.csv'}  ({len(store_df):,} rows)")
    print("  NOTE: synthetic data for testing only — replace with the real Kaggle CSVs.")


if __name__ == "__main__":
    main()
