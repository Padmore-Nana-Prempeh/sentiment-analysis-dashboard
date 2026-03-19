from __future__ import annotations

import os
from typing import Iterable, Optional

import pandas as pd

try:
    import praw
except Exception:
    praw = None


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


NUMERIC_LABEL_MAP = {
    0: "negative",
    1: "negative",
    2: "neutral",
    4: "positive",
}


def normalize_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    out = df.copy()
    if label_col not in out.columns:
        return out
    out[label_col] = out[label_col].map(lambda x: NUMERIC_LABEL_MAP.get(x, x))
    out[label_col] = out[label_col].astype(str).str.lower().str.strip()
    return out


def filter_by_keywords(df: pd.DataFrame, text_col: str, keywords: Iterable[str]) -> pd.DataFrame:
    keywords = [k.strip() for k in keywords if k and str(k).strip()]
    if not keywords:
        return df.copy()
    pattern = "|".join([rf"(?i)\b{k}\b" for k in keywords])
    return df[df[text_col].fillna("").str.contains(pattern, regex=True)].copy()


def collect_reddit_posts(
    query: str,
    subreddit: str = "all",
    limit: int = 1000,
    time_filter: str = "month",
) -> pd.DataFrame:
    if praw is None:
        raise ImportError("praw is not installed. Install it with pip install praw")

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "social-sentiment-app")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing Reddit credentials. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
        )

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    rows = []
    for submission in reddit.subreddit(subreddit).search(query, sort="new", time_filter=time_filter, limit=limit):
        rows.append(
            {
                "id": submission.id,
                "text": f"{submission.title} {submission.selftext}".strip(),
                "created_at": pd.to_datetime(submission.created_utc, unit="s"),
                "source": "reddit_submission",
                "score": submission.score,
                "subreddit": str(submission.subreddit),
                "brand": query,
            }
        )
    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
