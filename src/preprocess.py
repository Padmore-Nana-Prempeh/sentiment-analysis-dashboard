import re
import string
from typing import Iterable, List

import pandas as pd
import spacy

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#([A-Za-z0-9_]+)")
MULTISPACE_PATTERN = re.compile(r"\s+")


class TextPreprocessor:
    def __init__(self, model_name: str = "en_core_web_sm", batch_size: int = 256):
        self.batch_size = batch_size
        try:
            self.nlp = spacy.load(model_name, disable=["ner", "parser"])
        except OSError as exc:
            raise OSError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            ) from exc

    @staticmethod
    def clean_text(text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = URL_PATTERN.sub(" ", text)
        text = MENTION_PATTERN.sub(" ", text)
        text = HASHTAG_PATTERN.sub(r" \1 ", text)
        text = text.replace("&amp;", "and")
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = MULTISPACE_PATTERN.sub(" ", text).strip().lower()
        return text

    def lemmatize(self, texts: Iterable[str]) -> List[str]:
        docs = self.nlp.pipe(texts, batch_size=self.batch_size)
        processed = []
        for doc in docs:
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and not token.like_num and len(token) > 2
            ]
            processed.append(" ".join(tokens))
        return processed

    def transform_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        out = df.copy()
        out["clean_text"] = out[text_col].fillna("").astype(str).map(self.clean_text)
        out["processed_text"] = self.lemmatize(out["clean_text"].tolist())
        return out
