from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta

import pandas as pd

POSITIVE = [
    "I love the battery life on this {brand}",
    "The new {brand} update is amazing",
    "Great value and smooth experience with {brand}",
    "Customer support from {brand} was helpful",
    "The design quality of {brand} is excellent",
]

NEGATIVE = [
    "I hate the latest {brand} update",
    "The price of {brand} is too high",
    "Very disappointed with {brand} support",
    "The {brand} app keeps crashing",
    "Poor battery and bad performance from {brand}",
]

NEUTRAL = [
    "I bought a new {brand} device today",
    "The {brand} launch event happened this week",
    "Trying {brand} for the first time",
    "I saw a discussion about {brand} online",
    "Looking at {brand} pricing options",
]


def generate_labeled_dataset(n: int, brand: str) -> pd.DataFrame:
    rows = []
    start = datetime(2026, 1, 1)
    templates = [(POSITIVE, "positive"), (NEGATIVE, "negative"), (NEUTRAL, "neutral")]
    for i in range(n):
        choices, label = random.choice(templates)
        text = random.choice(choices).format(brand=brand)
        dt = start + timedelta(days=random.randint(0, 59))
        rows.append({"text": text, "label": label, "created_at": dt.isoformat(), "brand": brand})
    return pd.DataFrame(rows)


def generate_brand_posts(n: int, brand: str) -> pd.DataFrame:
    df = generate_labeled_dataset(n=n, brand=brand).drop(columns=["label"])
    df["source"] = [random.choice(["twitter", "reddit"]) for _ in range(len(df))]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo data for the project")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-posts", type=int, default=5000)
    parser.add_argument("--brand", type=str, default="Tesla")
    parser.add_argument("--outdir", type=str, default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    train_df = generate_labeled_dataset(args.n_train, args.brand)
    posts_df = generate_brand_posts(args.n_posts, args.brand)
    train_df.to_csv(os.path.join(args.outdir, "demo_training_data.csv"), index=False)
    posts_df.to_csv(os.path.join(args.outdir, "demo_brand_posts.csv"), index=False)
    print("Demo datasets created")


if __name__ == "__main__":
    main()
