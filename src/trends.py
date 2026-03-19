from __future__ import annotations

import pandas as pd


SENTIMENT_TO_SCORE = {
    "negative": -1,
    "neutral": 0,
    "positive": 1,
}


def prepare_time_series(df: pd.DataFrame, date_col: str = "created_at", sentiment_col: str = "predicted_sentiment", freq: str = "D") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).copy()
    out["sentiment_score"] = out[sentiment_col].map(SENTIMENT_TO_SCORE).fillna(0)
    out["period"] = out[date_col].dt.to_period(freq).dt.to_timestamp()

    agg = (
        out.groupby("period")
        .agg(
            posts=(sentiment_col, "size"),
            avg_sentiment=("sentiment_score", "mean"),
            positive_rate=(sentiment_col, lambda s: (s == "positive").mean()),
            negative_rate=(sentiment_col, lambda s: (s == "negative").mean()),
        )
        .reset_index()
        .sort_values("period")
    )
    return agg


def detect_trend_spikes(ts: pd.DataFrame, window: int = 7, z_threshold: float = 2.0) -> pd.DataFrame:
    out = ts.copy().sort_values("period")
    out["rolling_mean"] = out["avg_sentiment"].rolling(window=window, min_periods=3).mean()
    out["rolling_std"] = out["avg_sentiment"].rolling(window=window, min_periods=3).std()
    out["z_score"] = (out["avg_sentiment"] - out["rolling_mean"]) / out["rolling_std"]
    out["trend_flag"] = "normal"
    out.loc[out["z_score"] >= z_threshold, "trend_flag"] = "positive_spike"
    out.loc[out["z_score"] <= -z_threshold, "trend_flag"] = "negative_spike"
    return out
