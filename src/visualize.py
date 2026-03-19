from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


def save_sentiment_distribution(df: pd.DataFrame, sentiment_col: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    counts = df[sentiment_col].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_trend_line(ts: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(11, 5))
    plt.plot(ts["period"], ts["avg_sentiment"])
    plt.title("Average Sentiment Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_posts_volume(ts: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(11, 5))
    plt.plot(ts["period"], ts["posts"])
    plt.title("Post Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Posts")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_topic_wordcloud(text: str, output_path: str, title: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wc = WordCloud(width=1000, height=500, background_color="white").generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
