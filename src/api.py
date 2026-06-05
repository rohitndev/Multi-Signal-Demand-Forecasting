"""
api.py
------
FastAPI REST layer exposing the forecasting, replenishment, and drift features.

Run:
    uvicorn src.api:app --reload

Endpoints:
    POST /forecast   -> P10/P50/P90 forecast for a store
    POST /replenish  -> purchase order recommendation
    GET  /drift      -> drifted features + drift score + MAPE flag
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import predict as predict_mod
from src import agent as agent_mod
from src import drift as drift_mod

app = FastAPI(
    title="Multi-Signal Demand Forecasting System",
    description="TFT-based demand forecasting with replenishment agent and drift detection.",
    version="1.0.0",
)


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class ForecastRequest(BaseModel):
    store_id: int = Field(..., examples=[1])
    forecast_days: int = Field(28, ge=1, le=28)


class ReplenishRequest(BaseModel):
    store_id: int = Field(..., examples=[1])
    current_inventory: int = Field(..., ge=0, examples=[500])
    reorder_point: int = Field(100, ge=0, examples=[100])


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict:
    """Health check / index."""
    return {"status": "ok", "endpoints": ["/forecast", "/replenish", "/drift"]}


@app.post("/forecast")
def forecast(req: ForecastRequest) -> dict:
    """Return P10/P50/P90 forecasts for a store."""
    try:
        return predict_mod.forecast(req.store_id, req.forecast_days)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}")


@app.post("/replenish")
def replenish(req: ReplenishRequest) -> dict:
    """Forecast the store, then return a purchase order recommendation."""
    try:
        fc = predict_mod.forecast(req.store_id, 28)
        p50 = [row["p50"] for row in fc["forecasts"]]
        return agent_mod.replenish(
            store_id=req.store_id,
            p50_forecast=p50,
            current_inventory=req.current_inventory,
            reorder_point=req.reorder_point,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Replenish failed: {exc}")


@app.get("/drift")
def drift() -> dict:
    """Run the Evidently drift report over reference vs current windows."""
    try:
        return drift_mod.run()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Drift check failed: {exc}")
