"""
train.py
--------
Train a Temporal Fusion Transformer (TFT) on the engineered Rossmann data and
log P10/P50/P90 quantile forecasts plus MAPE/MAE to MLflow.

Run:
    python -m src.train

Notes for college use:
  * Defaults to 5 epochs on CPU.
  * By default trains on a small subset of stores so it finishes quickly.
    Pass --all-stores (or n_stores=None in train()) for the full dataset.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from src.config import CLEAN_CSV, MODEL_CKPT, MLRUNS_DIR
from src.features import build_features


def _prepare(n_stores: int | None = 20) -> pd.DataFrame:
    """Load clean data, engineer features, and add the integer time index TFT needs."""
    df = pd.read_csv(CLEAN_CSV)
    df["Date"] = pd.to_datetime(df["Date"])

    # Keep a manageable subset of stores so training is fast on a laptop.
    if n_stores is not None:
        keep = sorted(df["Store"].unique())[:n_stores]
        df = df[df["Store"].isin(keep)].copy()

    # tsfresh is skipped here: the static per-store features don't help the TFT
    # time-index model much and they slow training down. Flip to True if wanted.
    df = build_features(df, run_tsfresh=False)

    # TFT requires a contiguous integer time index per series.
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("Store").cumcount()

    # Target must be float: GroupNormalizer's softplus transform calls
    # torch.finfo(), which rejects integer dtypes.
    df["Sales"] = df["Sales"].astype("float32")

    # Categoricals must be strings for pytorch-forecasting.
    df["Store"] = df["Store"].astype(str)
    df["month"] = df["month"].astype(str)
    df["DayOfWeek"] = df["DayOfWeek"].astype(str)
    return df


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring zero-actual rows."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def train(epochs: int = 5, n_stores: int | None = 20) -> float:
    """Train the TFT, log to MLflow, save the checkpoint, return final MAPE."""
    import mlflow
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
    from pytorch_forecasting.metrics import QuantileLoss
    # Newer pytorch-forecasting builds models on the `lightning.pytorch` package,
    # so the Trainer must come from there too (not the legacy `pytorch_lightning`).
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping

    max_encoder_length = 28
    max_prediction_length = 28

    df = _prepare(n_stores=n_stores)

    # Train/validation split on the time index (last 28 days held out per series).
    training_cutoff = df["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="Sales",
        group_ids=["Store"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["Store"],
        time_varying_known_reals=["time_idx", "Promo", "promo_flag", "is_weekend"],
        time_varying_known_categoricals=["month", "DayOfWeek"],
        time_varying_unknown_reals=[
            "Sales", "sales_lag_7", "sales_rolling_mean_7",
        ],
        # add_nan=True lets the validation window contain calendar categories
        # (e.g. a month) that never appeared in the training window.
        categorical_encoders={
            "month": NaNLabelEncoder(add_nan=True),
            "DayOfWeek": NaNLabelEncoder(add_nan=True),
        },
        target_normalizer=GroupNormalizer(groups=["Store"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )

    batch_size = 64
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # ---- MLflow tracking (local SQLite backend) ----
    # Newer MLflow rejects the bare './mlruns' file store, so we use a local
    # SQLite DB (matches the project's "SQLite for local storage" requirement).
    # Artifacts (the checkpoint) still land under mlruns/.
    db_uri = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(db_uri)
    experiment = mlflow.get_experiment_by_name("ds02-tft-demand-forecasting")
    if experiment is None:
        mlflow.create_experiment(
            "ds02-tft-demand-forecasting",
            artifact_location=(MLRUNS_DIR / "artifacts").as_uri(),
        )
    mlflow.set_experiment("ds02-tft-demand-forecasting")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "n_stores": n_stores if n_stores is not None else "all",
                "max_encoder_length": max_encoder_length,
                "max_prediction_length": max_prediction_length,
                "quantiles": "[0.1, 0.5, 0.9]",
            }
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            log_interval=0,
            optimizer="adam",
        )

        trainer = Trainer(
            max_epochs=epochs,
            accelerator="cpu",
            gradient_clip_val=0.1,
            enable_progress_bar=True,
            enable_checkpointing=False,
            logger=False,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
        )

        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # ---- Evaluate on the validation set ----
        raw = tft.predict(val_loader, mode="raw", return_x=True)
        # P50 is the middle quantile (index 1 of [0.1, 0.5, 0.9]).
        preds_p50 = raw.output.prediction[..., 1].numpy().reshape(-1)
        actuals = torch.cat([y[0] for _, y in iter(val_loader)]).numpy().reshape(-1)

        mae = float(np.mean(np.abs(actuals - preds_p50)))
        mape = _mape(actuals, preds_p50)
        val_loss = float(trainer.callback_metrics.get("val_loss", torch.tensor(float("nan"))))

        mlflow.log_metrics({"MAPE": mape, "MAE": mae, "val_loss": val_loss})

        # ---- Save checkpoint ----
        trainer.save_checkpoint(str(MODEL_CKPT))
        mlflow.log_artifact(str(MODEL_CKPT))

    print("\nTraining complete.")
    print(f"  Final MAPE : {mape:.2f}%")
    print(f"  Final MAE  : {mae:.2f}")
    print(f"  Checkpoint : {MODEL_CKPT}")
    return mape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TFT demand model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--all-stores", action="store_true", help="Train on every store.")
    parser.add_argument("--n-stores", type=int, default=20)
    args = parser.parse_args()

    train(epochs=args.epochs, n_stores=None if args.all_stores else args.n_stores)
