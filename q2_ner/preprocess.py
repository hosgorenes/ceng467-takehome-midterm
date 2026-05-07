from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from q2_ner.config import Q2Config
from shared.seed import set_seed


@dataclass(frozen=True)
class LabelSpace:
    labels: list[str]
    label2id: dict[str, int]
    id2label: dict[int, str]


def load_conll(cfg: Q2Config = Q2Config()) -> tuple[DatasetDict, LabelSpace]:
    set_seed(cfg.seed)
    raw = load_dataset(cfg.dataset_name, revision="refs/convert/parquet")
    label_list = raw["train"].features["ner_tags"].feature.names
    ls = LabelSpace(
        labels=label_list,
        label2id={l: i for i, l in enumerate(label_list)},
        id2label={i: l for i, l in enumerate(label_list)},
    )
    return raw, ls


def tokenize_and_align_labels(
    dataset: DatasetDict,
    tokenizer_name: str,
    max_len: int,
) -> tuple[DatasetDict, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def _map(examples):
        tokenized = tok(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        all_labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev_wid:
                    label_ids.append(int(ner_tags[wid]))
                else:
                    label_ids.append(-100)
                prev_wid = wid
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    tokenized = dataset.map(_map, batched=True, remove_columns=dataset["train"].column_names)
    tokenized.set_format("torch")
    return tokenized, tok


def split_stats(dataset: DatasetDict) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for split in ["train", "validation", "test"]:
        n_sent = len(dataset[split])
        n_tok = sum(len(s) for s in dataset[split]["tokens"])
        stats[split] = {"sentences": float(n_sent), "tokens": float(n_tok)}
    return stats


def bert_token_len_stats(tok: AutoTokenizer, token_seqs: list[list[str]], n: int = 500) -> dict[str, float]:
    lens = [len(tok(s, is_split_into_words=True)["input_ids"]) for s in token_seqs[:n]]
    return {
        "mean": float(np.mean(lens)),
        "max": float(max(lens) if lens else 0),
        "p95": float(np.percentile(lens, 95)) if lens else 0.0,
    }

