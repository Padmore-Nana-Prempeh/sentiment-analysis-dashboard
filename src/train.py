from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from collect import normalize_labels
from preprocess import TextPreprocessor


VALID_LABELS = {"negative", "neutral", "positive"}


def prepare_training_data(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    out = normalize_labels(df, label_col=label_col)
    out = out[[text_col, label_col]].dropna().copy()
    out[label_col] = out[label_col].astype(str).str.lower().str.strip()
    out = out[out[label_col].isin(VALID_LABELS)].copy()
    return out


def train_sentiment_model(df: pd.DataFrame, text_col: str, label_col: str) -> Tuple[Pipeline, dict, pd.DataFrame, pd.DataFrame]:
    prep = TextPreprocessor()
    df = prep.transform_dataframe(df, text_col=text_col)

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"],
        df[label_col],
        test_size=0.2,
        random_state=42,
        stratify=df[label_col],
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    labels = ["negative", "neutral", "positive"]
    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels)
    return pipeline, metrics, report_df, cm_df


def save_artifacts(pipeline: Pipeline, metrics: dict, report_df: pd.DataFrame, cm_df: pd.DataFrame, model_dir: str, output_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["clf"]

    joblib.dump(pipeline, os.path.join(model_dir, "sentiment_pipeline.joblib"))
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(model, os.path.join(model_dir, "sentiment_model.joblib"))

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sentiment classifier")
    parser.add_argument("--input", required=True, help="Path to labeled CSV file")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = prepare_training_data(df, text_col=args.text_col, label_col=args.label_col)

    if df.empty:
        raise ValueError("No valid labeled rows found after preprocessing label values.")

    pipeline, metrics, report_df, cm_df = train_sentiment_model(df, text_col=args.text_col, label_col=args.label_col)
    save_artifacts(pipeline, metrics, report_df, cm_df, model_dir=args.model_dir, output_dir=args.output_dir)

    print("Training complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
