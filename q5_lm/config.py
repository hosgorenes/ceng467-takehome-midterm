from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Q5Config:
    seed: int = 42
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"

    # Vocabulary
    min_freq: int = 3

    # N-gram parameters
    ngram_n: int = 3
    ngram_alpha: float = 0.1  # Laplace smoothing
    ngram_eval_subset: int = 5000

    # LSTM parameters
    lstm_batch_size: int = 64
    lstm_seq_len: int = 35  # BPTT chunk size
    lstm_embed_dim: int = 512
    lstm_hidden_dim: int = 512
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.35
    lstm_lr: float = 5e-3
    lstm_epochs: int = 15
    lstm_grad_clip: float = 0.25
    lstm_tie_weights: bool = True

    # Generation
    gen_temperature: float = 0.8
    gen_n_words: int = 40
