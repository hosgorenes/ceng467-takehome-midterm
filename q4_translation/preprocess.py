from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from q4_translation.config import Q4Config
from shared.seed import set_seed

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"


@dataclass(frozen=True)
class TranslationData:
    train_src: list[str]
    train_tgt: list[str]
    val_src: list[str]
    val_tgt: list[str]
    test_src: list[str]
    test_tgt: list[str]
    test_src_raw: list[str]
    test_tgt_raw: list[str]


@dataclass
class Vocabs:
    src_vocab: dict[str, int]
    tgt_vocab: dict[str, int]
    inv_tgt: dict[int, str]


def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"([.!?,\"'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_multi30k(cfg: Q4Config = Q4Config()) -> TranslationData:
    set_seed(cfg.seed)
    raw = load_dataset(cfg.dataset_name)

    def get_pairs(split: str) -> tuple[list[str], list[str]]:
        src = [preprocess_text(ex["en"]) for ex in raw[split]]
        tgt = [preprocess_text(ex["de"]) for ex in raw[split]]
        return src, tgt

    train_src, train_tgt = get_pairs("train")
    val_src, val_tgt = get_pairs("validation")
    test_src, test_tgt = get_pairs("test")

    test_src_raw = [ex["en"] for ex in raw["test"]]
    test_tgt_raw = [ex["de"] for ex in raw["test"]]

    return TranslationData(
        train_src=train_src,
        train_tgt=train_tgt,
        val_src=val_src,
        val_tgt=val_tgt,
        test_src=test_src,
        test_tgt=test_tgt,
        test_src_raw=test_src_raw,
        test_tgt_raw=test_tgt_raw,
    )


def build_vocab(sentences: list[str], min_freq: int = 2) -> dict[str, int]:
    counter = Counter(w for s in sentences for w in s.split())
    vocab = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def build_vocabs(data: TranslationData, cfg: Q4Config = Q4Config()) -> Vocabs:
    src_vocab = build_vocab(data.train_src, cfg.min_freq)
    tgt_vocab = build_vocab(data.train_tgt, cfg.min_freq)
    inv_tgt = {i: w for w, i in tgt_vocab.items()}
    return Vocabs(src_vocab=src_vocab, tgt_vocab=tgt_vocab, inv_tgt=inv_tgt)


def encode_sentence(sentence: str, vocab: dict[str, int], max_len: int = 50) -> list[int]:
    ids = [vocab.get(SOS, 2)]
    ids += [vocab.get(w, vocab[UNK]) for w in sentence.split()[:max_len]]
    ids += [vocab.get(EOS, 3)]
    return ids


class TranslationDataset(Dataset):
    def __init__(
        self,
        srcs: list[str],
        tgts: list[str],
        src_vocab: dict[str, int],
        tgt_vocab: dict[str, int],
        max_len: int = 50,
    ):
        self.pairs = [
            (encode_sentence(s, src_vocab, max_len), encode_sentence(t, tgt_vocab, max_len))
            for s, t in zip(srcs, tgts)
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = pad_sequence([torch.tensor(s) for s in srcs], batch_first=True, padding_value=0)
    tgts = pad_sequence([torch.tensor(t) for t in tgts], batch_first=True, padding_value=0)
    return srcs, tgts


def create_dataloaders(
    data: TranslationData,
    vocabs: Vocabs,
    cfg: Q4Config = Q4Config(),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TranslationDataset(data.train_src, data.train_tgt, vocabs.src_vocab, vocabs.tgt_vocab, cfg.max_len)
    val_ds = TranslationDataset(data.val_src, data.val_tgt, vocabs.src_vocab, vocabs.tgt_vocab, cfg.max_len)
    test_ds = TranslationDataset(data.test_src, data.test_tgt, vocabs.src_vocab, vocabs.tgt_vocab, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.seq_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.seq_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.seq_batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def dataset_stats(data: TranslationData) -> dict[str, dict[str, float]]:
    src_lens = [len(s.split()) for s in data.train_src]
    tgt_lens = [len(t.split()) for t in data.train_tgt]

    return {
        "train_size": len(data.train_src),
        "val_size": len(data.val_src),
        "test_size": len(data.test_src),
        "en_tokens": {
            "mean": float(np.mean(src_lens)),
            "max": int(max(src_lens)) if src_lens else 0,
        },
        "de_tokens": {
            "mean": float(np.mean(tgt_lens)),
            "max": int(max(tgt_lens)) if tgt_lens else 0,
        },
    }
