from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import precision_score, recall_score


@dataclass(frozen=True)
class NERMetrics:
    precision: float
    recall: float
    f1: float


def compute_ner_metrics(true_seqs: list[list[str]], pred_seqs: list[list[str]]) -> NERMetrics:
    return NERMetrics(
        precision=float(precision_score(true_seqs, pred_seqs)),
        recall=float(recall_score(true_seqs, pred_seqs)),
        f1=float(seq_f1(true_seqs, pred_seqs)),
    )


def analyze_boundary_and_confusion_errors(
    true_seqs: list[list[str]],
    pred_seqs: list[list[str]],
    token_seqs: list[list[str]],
    limit_each: int = 5,
) -> dict[str, Any]:
    boundary = []
    confusion = []
    for t_seq, p_seq, toks in zip(true_seqs, pred_seqs, token_seqs):
        for t, p, tok in zip(t_seq, p_seq, toks):
            if t == p:
                continue
            t_type = t.split("-")[-1] if "-" in t else t
            p_type = p.split("-")[-1] if "-" in p else p
            if t_type == p_type and t != p:
                if len(boundary) < limit_each:
                    boundary.append({"token": tok, "true": t, "pred": p})
            elif t != "O" and p != "O" and t_type != p_type:
                if len(confusion) < limit_each:
                    confusion.append({"token": tok, "true": t, "pred": p})
        if len(boundary) >= limit_each and len(confusion) >= limit_each:
            break

    return {"boundary_examples": boundary, "confusion_examples": confusion}

