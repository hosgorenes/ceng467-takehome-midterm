from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Q1Config:
    seed: int = 42
    dataset_name: str = "imdb"

    # Splits: HF imdb already has train/test; we split train into train/val
    val_size: float = 0.2

    # TF-IDF
    tfidf_max_features_uni: int = 30_000
    tfidf_max_features_bi: int = 50_000
    tfidf_min_df_uni: int = 2
    tfidf_min_df_bi: int = 3

    # BiLSTM
    bilstm_max_vocab: int = 30_000
    bilstm_max_len: int = 256

    # DistilBERT
    bert_model: str = "distilbert-base-uncased"
    bert_max_len: int = 256

