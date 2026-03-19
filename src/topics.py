from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


DEFAULT_TOPIC_MAP = {
    "price": "pricing",
    "pricing": "pricing",
    "cost": "pricing",
    "expensive": "pricing",
    "cheap": "pricing",
    "quality": "quality",
    "build": "quality",
    "design": "quality",
    "feature": "features",
    "features": "features",
    "camera": "features",
    "battery": "features",
    "support": "support",
    "service": "support",
    "delivery": "operations",
    "shipping": "operations",
    "app": "software",
    "update": "software",
    "bug": "software",
}


def fit_lda_topics(
    texts: List[str],
    n_topics: int = 8,
    max_features: int = 5000,
    random_state: int = 42,
) -> Tuple[LatentDirichletAllocation, CountVectorizer, pd.DataFrame]:
    vectorizer = CountVectorizer(max_features=max_features, min_df=5, max_df=0.95, stop_words="english")
    dtm = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state, learning_method="batch")
    lda.fit(dtm)

    keywords = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-10:][::-1]
        words = [feature_names[i] for i in top_indices]
        category = map_topic_to_category(words)
        keywords.append(
            {
                "topic_id": topic_idx,
                "category": category,
                "keywords": ", ".join(words),
            }
        )
    keywords_df = pd.DataFrame(keywords)
    return lda, vectorizer, keywords_df


def assign_topics(lda: LatentDirichletAllocation, vectorizer: CountVectorizer, texts: List[str]) -> pd.DataFrame:
    dtm = vectorizer.transform(texts)
    topic_scores = lda.transform(dtm)
    topic_ids = topic_scores.argmax(axis=1)
    max_scores = topic_scores.max(axis=1)
    return pd.DataFrame({"topic_id": topic_ids, "topic_confidence": max_scores})


def map_topic_to_category(words: List[str]) -> str:
    for word in words:
        if word.lower() in DEFAULT_TOPIC_MAP:
            return DEFAULT_TOPIC_MAP[word.lower()]
    return "general_feedback"
