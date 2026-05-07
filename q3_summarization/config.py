from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Q3Config:
    seed: int = 42
    dataset_name: str = "cnn_dailymail"
    dataset_version: str = "3.0.0"

    # Subset sizes (full dataset too large for quick experiments)
    n_test: int = 500

    # TextRank
    textrank_n_sentences: int = 3
    textrank_language: str = "english"

    # BART
    bart_model: str = "facebook/bart-large-cnn"
    bart_max_input_words: int = 700
    bart_max_output_length: int = 128
    bart_min_output_length: int = 30
    bart_num_beams: int = 4
    bart_batch_size: int = 8

    # BERTScore subset (full is slow)
    bertscore_subset: int = 200
