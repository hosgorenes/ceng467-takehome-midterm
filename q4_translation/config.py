from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Q4Config:
    seed: int = 42
    dataset_name: str = "bentrevett/multi30k"

    # Vocabulary
    min_freq: int = 2
    max_len: int = 50

    # Seq2Seq hyperparameters
    seq_embed_dim: int = 256
    seq_hidden_dim: int = 512
    seq_num_layers: int = 2
    seq_dropout: float = 0.3
    seq_lr: float = 5e-4
    seq_epochs: int = 15
    seq_batch_size: int = 128
    seq_grad_clip: float = 1.0
    seq_tf_ratio: float = 0.5  # teacher forcing ratio

    # MarianMT
    marian_model: str = "Helsinki-NLP/opus-mt-en-de"
    marian_batch_size: int = 32

    # BERTScore subset
    bertscore_subset: int = 200
