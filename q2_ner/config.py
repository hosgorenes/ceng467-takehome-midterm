from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Q2Config:
    seed: int = 42
    dataset_name: str = "conll2003"
    bert_model: str = "bert-base-cased"
    max_len: int = 128

