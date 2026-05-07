from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from q1_classification.config import Q1Config
from shared.seed import set_seed


@dataclass(frozen=True)
class Q1Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    # Local import: nltk is optional unless stopwords are used.
    stop_words: set[str] = set()
    if remove_stopwords:
        import nltk
        from nltk.corpus import stopwords

        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))

    text = re.sub(r"<[^>]+>", " ", text)  # HTML tags
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)  # URLs
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\d+", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()

    if remove_stopwords:
        text = " ".join(w for w in text.split() if w not in stop_words)
    return text


def add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["clean"] = out["text"].apply(lambda x: clean_text(x, remove_stopwords=False))
    out["clean_no_stop"] = out["text"].apply(lambda x: clean_text(x, remove_stopwords=True))
    return out


def load_imdb_splits(cfg: Q1Config = Q1Config()) -> Q1Splits:
    set_seed(cfg.seed)
    raw = load_dataset(cfg.dataset_name)

    train_raw = pd.DataFrame({"text": raw["train"]["text"], "label": raw["train"]["label"]})
    test_raw = pd.DataFrame({"text": raw["test"]["text"], "label": raw["test"]["label"]})

    train_df, val_df = train_test_split(
        train_raw,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        stratify=train_raw["label"],
    )

    return Q1Splits(
        train=add_clean_columns(train_df.reset_index(drop=True)),
        val=add_clean_columns(val_df.reset_index(drop=True)),
        test=add_clean_columns(test_raw.reset_index(drop=True)),
    )


def token_stats(texts: Iterable[str]) -> dict[str, float | int]:
    lens = [len(t.split()) for t in texts]
    vocab_size = len(set(w for t in texts for w in t.split()))
    return {
        "mean_tokens": float(np.mean(lens)),
        "median_tokens": float(np.median(lens)),
        "max_tokens": int(max(lens) if lens else 0),
        "vocab_size": int(vocab_size),
    }

