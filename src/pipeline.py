from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd

from preprocess import TextPreprocessor
from topics import assign_topics, fit_lda_topics
from trends import detect_trend_spikes, prepare_time_series
from visualize import save_posts_volume, save_sentiment_distribution, save_topic_wordcloud, save_trend_line


def run_pipeline(posts_path: str, model_path: str, output_dir: str, date_col: str = "created_at", brand_col: str = "brand") -> None:
    os.makedirs(output_dir, exist_ok=True)
    posts = pd.read_csv(posts_path)

    prep = TextPreprocessor()
    posts = prep.transform_dataframe(posts, text_col="text")

    pipeline = joblib.load(model_path)
    posts["predicted_sentiment"] = pipeline.predict(posts["processed_text"])

    lda, topic_vectorizer, keywords_df = fit_lda_topics(posts["processed_text"].fillna("").tolist(), n_topics=8)
    assignments = assign_topics(lda, topic_vectorizer, posts["processed_text"].fillna("").tolist())
    posts = pd.concat([posts.reset_index(drop=True), assignments.reset_index(drop=True)], axis=1)

    ts = prepare_time_series(posts, date_col=date_col, sentiment_col="predicted_sentiment", freq="D")
    spikes = detect_trend_spikes(ts, window=7, z_threshold=2.0)

    posts.to_csv(os.path.join(output_dir, "predicted_posts.csv"), index=False)
    keywords_df.to_csv(os.path.join(output_dir, "topic_keywords.csv"), index=False)
    posts[["text", "topic_id", "topic_confidence"]].to_csv(os.path.join(output_dir, "topic_assignments.csv"), index=False)
    ts.to_csv(os.path.join(output_dir, "daily_sentiment.csv"), index=False)
    spikes.to_csv(os.path.join(output_dir, "trend_spikes.csv"), index=False)

    save_sentiment_distribution(posts, sentiment_col="predicted_sentiment", output_path=os.path.join(output_dir, "sentiment_distribution.png"))
    save_trend_line(ts, output_path=os.path.join(output_dir, "sentiment_trend.png"))
    save_posts_volume(ts, output_path=os.path.join(output_dir, "post_volume.png"))

    for _, row in keywords_df.iterrows():
        topic_id = int(row["topic_id"])
        topic_text = " ".join(posts.loc[posts["topic_id"] == topic_id, "processed_text"].dropna().tolist())
        if topic_text.strip():
            save_topic_wordcloud(
                topic_text,
                output_path=os.path.join(output_dir, f"topic_{topic_id}_wordcloud.png"),
                title=f"Topic {topic_id}: {row['category']}",
            )

    if brand_col in posts.columns:
        summary = (
            posts.groupby([brand_col, "predicted_sentiment"]).size().reset_index(name="count")
            .sort_values([brand_col, "count"], ascending=[True, False])
        )
        summary.to_csv(os.path.join(output_dir, "brand_sentiment_summary.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full sentiment and trend pipeline")
    parser.add_argument("--posts", required=True, help="CSV of brand posts")
    parser.add_argument("--model-path", required=True, help="Path to saved sentiment_pipeline.joblib")
    parser.add_argument("--vectorizer-path", required=False, help="Unused but kept for compatibility")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--date-col", default="created_at")
    parser.add_argument("--brand-col", default="brand")
    args = parser.parse_args()

    run_pipeline(
        posts_path=args.posts,
        model_path=args.model_path,
        output_dir=args.output_dir,
        date_col=args.date_col,
        brand_col=args.brand_col,
    )
    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
