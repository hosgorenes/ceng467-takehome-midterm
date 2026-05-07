from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset

from q5_lm.config import Q5Config
from shared.seed import set_seed

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"


@dataclass
class LMData:
    train_tokens: list[str]
    val_tokens: list[str]
    test_tokens: list[str]
    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]
    vocab: dict[str, int]
    inv_vocab: dict[int, str]


def get_tokens(dataset, split: str, lower: bool = True) -> list[str]:
    tokens = []
    for line in dataset[split]["text"]:
        line = line.strip()
        if not line or line.startswith(" ="):
            continue
        if lower:
            line = line.lower()
        tokens.extend(line.split())
    return tokens


def build_vocab(tokens: list[str], min_freq: int = 3) -> dict[str, int]:
    counter = Counter(tokens)
    vocab = {PAD: 0, UNK: 1, EOS: 2}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    return [vocab.get(t, vocab[UNK]) for t in tokens]


def load_wikitext(cfg: Q5Config = Q5Config()) -> LMData:
    set_seed(cfg.seed)
    raw = load_dataset(cfg.dataset_name, cfg.dataset_config)

    train_tokens = get_tokens(raw, "train")
    val_tokens = get_tokens(raw, "validation")
    test_tokens = get_tokens(raw, "test")

    vocab = build_vocab(train_tokens, cfg.min_freq)
    inv_vocab = {i: w for w, i in vocab.items()}

    train_ids = encode_tokens(train_tokens, vocab)
    val_ids = encode_tokens(val_tokens, vocab)
    test_ids = encode_tokens(test_tokens, vocab)

    return LMData(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        vocab=vocab,
        inv_vocab=inv_vocab,
    )


def batchify(data: list[int], batch_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(data, dtype=torch.long)
    n_batch = tensor.size(0) // batch_size
    tensor = tensor[: n_batch * batch_size]
    return tensor.view(batch_size, -1).t().contiguous().to(device)


def get_batch(source: torch.Tensor, i: int, seq_len: int):
    length = min(seq_len, source.size(0) - 1 - i)
    data = source[i : i + length]
    target = source[i + 1 : i + 1 + length].reshape(-1)
    return data, target


def dataset_stats(data: LMData) -> dict:
    oov_count = sum(1 for t in data.train_tokens if t not in data.vocab)
    oov_rate = oov_count / len(data.train_tokens) if data.train_tokens else 0

    return {
        "train_tokens": len(data.train_tokens),
        "val_tokens": len(data.val_tokens),
        "test_tokens": len(data.test_tokens),
        "vocab_size": len(data.vocab),
        "oov_rate": round(oov_rate * 100, 2),
    }
