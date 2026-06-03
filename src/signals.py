"""
signals.py
----------
Fetch exogenous demand signals and merge them onto the sales DataFrame.

Signals:
  * weather()  -> OpenWeatherMap current weather for Berlin (Rossmann is German)
  * trends()   -> Google Trends interest for the keyword "Rossmann"
  * sentiment()-> FinBERT average sentiment over a list of news headlines
  * build_signal_features(df) -> merge them all into the sales frame

Every external call is wrapped in try/except and falls back to mock / zero
values, so the whole project still runs fully offline.

Run a quick smoke test:
    python -m src.signals
"""
from __future__ import annotations

import pandas as pd

from src.config import get_env

# Default city for weather (Rossmann is a German drugstore chain).
DEFAULT_CITY = "Berlin"

# FinBERT model is loaded lazily so importing this module stays cheap.
_FINBERT = None


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def weather(city: str = DEFAULT_CITY) -> dict:
    """
    Return today's weather as {temp, rain, humidity}.
    Falls back to neutral mock values if the API key or network is unavailable.
    """
    mock = {"temp": 15.0, "rain": 0.0, "humidity": 60.0}
    api_key = get_env("OPENWEATHER_API_KEY")
    if not api_key:
        print("[weather] No OPENWEATHER_API_KEY set -> using mock values.")
        return mock

    try:
        import requests

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "temp": float(data["main"]["temp"]),
            # "rain" key only exists when it is actually raining.
            "rain": float(data.get("rain", {}).get("1h", 0.0)),
            "humidity": float(data["main"]["humidity"]),
        }
    except Exception as exc:  # noqa: BLE001 - intentional broad fallback
        print(f"[weather] API call failed ({exc}) -> using mock values.")
        return mock


# --------------------------------------------------------------------------- #
# Google Trends
# --------------------------------------------------------------------------- #
def trends(keyword: str = "Rossmann", days: int = 90) -> pd.DataFrame:
    """
    Pull Google Trends interest for `keyword` over the last `days` days.
    Returns a DataFrame with columns [date, trend]. On failure returns a
    zero-filled frame so callers always get the same shape.
    """
    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=0)
        timeframe = f"today {max(1, days // 30)}-m"  # e.g. "today 3-m"
        pytrends.build_payload([keyword], timeframe=timeframe)
        data = pytrends.interest_over_time()
        if data.empty:
            raise ValueError("empty Trends response")
        out = data.reset_index()[["date", keyword]]
        out.columns = ["date", "trend"]
        out["date"] = pd.to_datetime(out["date"])
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[trends] Fetch failed ({exc}) -> returning zero-filled trend.")
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
        return pd.DataFrame({"date": idx, "trend": 0.0})


# --------------------------------------------------------------------------- #
# News sentiment (FinBERT)
# --------------------------------------------------------------------------- #
def _load_finbert():
    """Lazily build the FinBERT sentiment pipeline (downloaded on first use)."""
    global _FINBERT
    if _FINBERT is None:
        from transformers import pipeline

        _FINBERT = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
        )
    return _FINBERT


def sentiment(headlines: list[str]) -> float:
    """
    Run FinBERT over a list of headlines and return one averaged score in
    [-1, 1]: positive -> +prob, negative -> -prob, neutral -> 0.
    Returns 0.0 if the model can't be loaded (offline).
    """
    if not headlines:
        return 0.0
    try:
        clf = _load_finbert()
        results = clf(headlines)
        score_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        scores = [score_map.get(r["label"].lower(), 0.0) * r["score"] for r in results]
        return float(sum(scores) / len(scores))
    except Exception as exc:  # noqa: BLE001
        print(f"[sentiment] FinBERT unavailable ({exc}) -> returning 0.0.")
        return 0.0


# --------------------------------------------------------------------------- #
# Merge everything into the sales frame
# --------------------------------------------------------------------------- #
def build_signal_features(
    df: pd.DataFrame, headlines: list[str] | None = None
) -> pd.DataFrame:
    """
    Add weather, Google Trends, and a news-sentiment column to `df`, keyed by
    date where possible. Missing signals are filled with 0 so training never
    breaks.

    Expects `df` to contain a 'Date' column.
    """
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])

    # --- Weather: a single snapshot, broadcast across all rows. ---
    w = weather()
    out["weather_temp"] = w["temp"]
    out["weather_rain"] = w["rain"]
    out["weather_humidity"] = w["humidity"]

    # --- Trends: merge on calendar date, fill gaps with 0. ---
    tr = trends()
    tr["date"] = pd.to_datetime(tr["date"]).dt.normalize()
    out = out.merge(
        tr.rename(columns={"date": "Date"}), on="Date", how="left"
    )
    out["trend"] = out["trend"].fillna(0.0)

    # --- Sentiment: one scalar broadcast across all rows. ---
    out["news_sentiment"] = sentiment(headlines or [])

    return out


if __name__ == "__main__":
    # Smoke test with a tiny synthetic frame.
    demo = pd.DataFrame(
        {"Date": pd.date_range("2015-07-01", periods=5, freq="D"), "Sales": range(5)}
    )
    print("weather() ->", weather())
    print("sentiment() ->", sentiment(["Rossmann reports record quarterly profit"]))
    enriched = build_signal_features(demo, headlines=["Strong sales growth at Rossmann"])
    print(enriched.head())
