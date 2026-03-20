from __future__ import annotations

import os
import json
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer


st.set_page_config(page_title="Sentiment & Trend Dashboard", layout="wide")
st.title("Social Media Sentiment & Market Trend Analysis")
st.caption("Consumer review sentiment, topic discovery, and brand-level trend monitoring from Amazon product reviews.")

outputs_dir = st.sidebar.text_input("Outputs directory", value="outputs")
pred_path = os.path.join(outputs_dir, "predicted_posts.csv")
ts_path = os.path.join(outputs_dir, "daily_sentiment.csv")
topic_path = os.path.join(outputs_dir, "topic_keywords.csv")

if not all(os.path.exists(p) for p in [pred_path, ts_path, topic_path]):
    st.warning("Run the pipeline first so the dashboard can load output files.")
    st.stop()

pred = pd.read_csv(pred_path)
ts = pd.read_csv(ts_path)
topics = pd.read_csv(topic_path) 

if "created_at" in pred.columns:
    pred["created_at"] = pd.to_datetime(pred["created_at"], errors="coerce")

if "period" in ts.columns:
    ts["period"] = pd.to_datetime(ts["period"], errors="coerce")
    ts = ts.sort_values("period").copy()

if "avg_sentiment" in ts.columns:
    ts["avg_sentiment_smooth"] = ts["avg_sentiment"].rolling(7, min_periods=1).mean()

if "posts" in ts.columns:
    ts["posts_smooth"] = ts["posts"].rolling(7, min_periods=1).mean()

sentiment_counts = pred["predicted_sentiment"].value_counts() if "predicted_sentiment" in pred.columns else pd.Series(dtype=int)
top_sentiment = sentiment_counts.idxmax() if not sentiment_counts.empty else "N/A"

pos_share = (pred["predicted_sentiment"].eq("positive").mean() * 100) if "predicted_sentiment" in pred.columns and len(pred) else 0
neg_share = (pred["predicted_sentiment"].eq("negative").mean() * 100) if "predicted_sentiment" in pred.columns and len(pred) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews analyzed", f"{len(pred):,}")
c2.metric("Unique topics", int(topics["topic_id"].nunique()) if "topic_id" in topics.columns else 0)
c3.metric("Brands", int(pred["brand"].nunique()) if "brand" in pred.columns else 1)
c4.metric("Dominant sentiment", top_sentiment.title() if isinstance(top_sentiment, str) else "N/A")


metrics_path = os.path.join("models", "metrics.json")

if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)

    st.subheader("Model performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    m3.metric("Weighted F1", f"{metrics['weighted_f1']:.3f}")


if sentiment_counts.size > 0:
    st.success(f"Overall sentiment is strongly {top_sentiment.lower()} with {pos_share:.1f}% positive reviews.")


st.subheader("🚨 Alerts")

if "predicted_sentiment" in pred.columns:
    neg_share = pred["predicted_sentiment"].eq("negative").mean() * 100 if len(pred) else 0

    if neg_share > 20:
        st.error(f"High negative sentiment detected ({neg_share:.1f}%)")

if "posts" in ts.columns and not ts.empty:
    if ts["posts"].max() > ts["posts"].mean() * 3:
        st.warning("Unusual spike in review activity detected.")




st.subheader("Key insights")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    if "predicted_sentiment" in pred.columns:
        st.markdown(
            f"""
- Positive sentiment share: **{pos_share:.1f}%**
- Negative sentiment share: **{neg_share:.1f}%**
- Most common sentiment class: **{top_sentiment.title()}**
"""
        )

with insight_col2:
    if "avg_sentiment" in ts.columns and "period" in ts.columns and not ts.empty:
        best_idx = ts["avg_sentiment"].idxmax()
        worst_idx = ts["avg_sentiment"].idxmin()
        best_period = ts.loc[best_idx, "period"]
        worst_period = ts.loc[worst_idx, "period"]

        best_period_str = best_period.strftime("%Y-%m-%d") if pd.notna(best_period) else "N/A"
        worst_period_str = worst_period.strftime("%Y-%m-%d") if pd.notna(worst_period) else "N/A"

        st.markdown(
            f"""
- Highest average sentiment period: **{best_period_str}**
- Lowest average sentiment period: **{worst_period_str}**
"""
        )

st.divider()

st.subheader("Sentiment distribution")
if not sentiment_counts.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {
        "positive": "green",
        "neutral": "gray",
        "negative": "red",
    }
    sentiment_counts = sentiment_counts.reindex(
        [c for c in ["negative", "neutral", "positive"] if c in sentiment_counts.index]
        + [c for c in sentiment_counts.index if c not in ["negative", "neutral", "positive"]]
    )
    sentiment_counts.plot(
        kind="bar",
        color=[colors.get(x, "steelblue") for x in sentiment_counts.index],
        ax=ax,
    )
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Review count")
    ax.set_title("Sentiment Distribution")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
else:
    st.info("No sentiment predictions available.")

st.divider()

trend_col1, trend_col2 = st.columns(2)

with trend_col1:
    st.subheader("Average sentiment over time")
    sentiment_trend_path = os.path.join(outputs_dir, "sentiment_trend.png")
    if os.path.exists(sentiment_trend_path):
        st.image(sentiment_trend_path, use_container_width=True)
    else:
        st.info("No sentiment trend chart available.")

with trend_col2:
    st.subheader("Post volume over time")
    post_volume_path = os.path.join(outputs_dir, "post_volume.png")
    if os.path.exists(post_volume_path):
        st.image(post_volume_path, use_container_width=True)
    else:
        st.info("No post volume chart available.")

st.divider()

if "brand" in pred.columns:
    st.subheader("Brand sentiment ranking")

    brand_sentiment = (
        pred.groupby("brand")["predicted_sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["negative", "neutral", "positive"]:
        if col not in brand_sentiment.columns:
            brand_sentiment[col] = 0.0

    brand_sentiment["net_sentiment"] = brand_sentiment["positive"] - brand_sentiment["negative"]
    brand_sentiment = brand_sentiment.sort_values("net_sentiment", ascending=False)

    display_brand_sentiment = brand_sentiment.copy()
    for col in ["negative", "neutral", "positive", "net_sentiment"]:
        display_brand_sentiment[col] = display_brand_sentiment[col].round(3)

    st.dataframe(display_brand_sentiment, width="stretch")

st.divider()

if "name" in pred.columns and "predicted_sentiment" in pred.columns:
    st.subheader("Top products with negative sentiment")

    negative_products = pred[pred["predicted_sentiment"] == "negative"]
    if not negative_products.empty:
        worst_products = (
            negative_products.groupby("name")
            .size()
            .reset_index(name="negative_review_count")
            .sort_values("negative_review_count", ascending=False)
            .head(10)
        )
        st.dataframe(worst_products, width="stretch")
    else:
        st.info("No negative product reviews found in the current filtered dataset.")

        st.subheader("Top complaint phrases")

        negative_text = pred[pred["predicted_sentiment"] == "negative"]["text"]

    if len(negative_text) > 0:
        text_blob = " ".join(negative_text.astype(str)).lower()
        words = re.findall(r"\b[a-z]{4,}\b", text_blob)

        custom_stopwords = set(ENGLISH_STOP_WORDS).union(
            {
                "amazon", "product", "item", "would", "could",
                "really", "still", "well", "using", "bought",
                "just", "like", "good", "great", "time", "work",
                "don", "didn", "doesn", "isn", "wasn",
                "brand", "device", "thing",
                "alkaline", "amazonbasics", "basic", "pack",
                "count", "aaa", "aa", "used", "lasted"
            }
        )

        filtered_words = [w for w in words if w not in custom_stopwords]
        cleaned_text = " ".join(filtered_words)

        vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=30)
        X = vectorizer.fit_transform([cleaned_text])

        phrase_counts = pd.Series(
            X.toarray()[0],
            index=vectorizer.get_feature_names_out()
        ).sort_values(ascending=False)

        clean_phrases = {}

        for phrase, count in phrase_counts.items():
            words = phrase.split()

            normalized = []
            for w in words:
                if w.endswith("ies") and len(w) > 4:
                    w = w[:-3] + "y"
                elif w.endswith("s") and len(w) > 4:
                    w = w[:-1]
                normalized.append(w)
            words = normalized

            key = " ".join(words)
            clean_phrases[key] = clean_phrases.get(key, 0) + count

        final_phrase_counts = (
            pd.Series(clean_phrases)
            .sort_values(ascending=False)
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        final_phrase_counts.plot(kind="bar", ax=ax)
        ax.set_title("Top Complaint Phrases")
        ax.set_xlabel("Phrase")
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No negative complaints found.")



st.divider()

st.subheader("Extracted topics")
st.dataframe(topics, width="stretch")

st.divider()

if "brand" in pred.columns:
    st.subheader("Brand filter")
    brands = ["All"] + sorted(pred["brand"].dropna().astype(str).unique().tolist())
    choice = st.selectbox("Select brand", brands)

    view = pred if choice == "All" else pred[pred["brand"].astype(str) == choice]

    if "predicted_sentiment" in view.columns:
        if len(view) > 0:
            brand_neg_share = view["predicted_sentiment"].eq("negative").mean() * 100
            st.markdown(f"Filtered reviews: **{len(view):,}** | Negative share: **{brand_neg_share:.1f}%**")
        else:
            st.markdown("No records available for this brand.")

    cols = [
        c for c in view.columns
        if c in ["brand", "name", "text", "predicted_sentiment", "topic_id", "topic_confidence", "created_at"]
    ]
    st.dataframe(view[cols], width="stretch")
else:
    st.subheader("Scored posts")
    cols = [
        c for c in pred.columns
        if c in ["name", "text", "predicted_sentiment", "topic_id", "topic_confidence", "created_at"]
    ]
    st.dataframe(pred[cols], width="stretch")











