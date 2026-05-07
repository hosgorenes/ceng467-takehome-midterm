from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from q3_summarization.config import Q3Config
from shared.io import read_json, write_json
from shared.paths import get_paths

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class SummarizationMetrics:
    rouge1: float
    rouge2: float
    rougeL: float
    bleu: float
    meteor: float
    bertscore_f1: float


def compute_summarization_metrics(
    predictions: list[str],
    references: list[str],
    bertscore_subset: int = 200,
    device: str | None = None,
) -> SummarizationMetrics:
    import evaluate
    from bert_score import score as bert_score_fn

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ROUGE
    rouge = evaluate.load("rouge")
    r = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    # BLEU
    bleu = evaluate.load("bleu")
    bleu_preds = [p.split() for p in predictions]
    bleu_refs = [[ref.split()] for ref in references]
    b = bleu.compute(predictions=bleu_preds, references=bleu_refs)

    # METEOR
    meteor = evaluate.load("meteor")
    m = meteor.compute(predictions=predictions, references=references)

    # BERTScore (subset for speed)
    n = min(bertscore_subset, len(predictions))
    P, R, F1 = bert_score_fn(
        predictions[:n],
        references[:n],
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
        device=dev,
    )

    return SummarizationMetrics(
        rouge1=float(r["rouge1"]) * 100,
        rouge2=float(r["rouge2"]) * 100,
        rougeL=float(r["rougeL"]) * 100,
        bleu=float(b["bleu"]) * 100,
        meteor=float(m["meteor"]) * 100,
        bertscore_f1=float(F1.mean().item()) * 100,
    )


def qualitative_examples(
    articles: list[str],
    references: list[str],
    predictions: list[str],
    indices: list[int] | None = None,
    n_words: int = 200,
) -> list[dict[str, Any]]:
    idxs = indices or [0, 7, 42]
    examples = []
    for idx in idxs:
        if idx < len(articles):
            examples.append({
                "idx": idx,
                "article_excerpt": " ".join(articles[idx].split()[:n_words]),
                "reference": references[idx],
                "prediction": predictions[idx],
            })
    return examples


def avg_summary_length(summaries: list[str]) -> float:
    return float(np.mean([len(s.split()) for s in summaries]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--textrank", type=str, default="", help="Path to textrank summaries.json")
    ap.add_argument("--bart", type=str, default="", help="Path to bart summaries.json")
    ap.add_argument("--out", type=str, default="", help="Output directory")
    args = ap.parse_args()

    cfg = Q3Config()
    artifacts = get_paths().artifacts / "q3"

    textrank_path = Path(args.textrank) if args.textrank else artifacts / "textrank" / "summaries.json"
    bart_path = Path(args.bart) if args.bart else artifacts / "bart" / "summaries.json"

    results: dict[str, Any] = {"models": {}}

    for name, path in [("textrank", textrank_path), ("bart", bart_path)]:
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue

        print(f"\nEvaluating {name}...")
        data = read_json(path)
        preds = data["summaries"]
        refs = data["references"]

        m = compute_summarization_metrics(preds, refs, bertscore_subset=cfg.bertscore_subset)

        results["models"][name] = {
            "metrics": {
                "ROUGE-1": round(m.rouge1, 2),
                "ROUGE-2": round(m.rouge2, 2),
                "ROUGE-L": round(m.rougeL, 2),
                "BLEU": round(m.bleu, 2),
                "METEOR": round(m.meteor, 2),
                "BERTScore-F1": round(m.bertscore_f1, 2),
            },
            "avg_length": round(avg_summary_length(preds), 1),
            "qualitative": qualitative_examples(
                data.get("articles", [""] * len(preds)),
                refs,
                preds,
            ),
        }

    out_dir = Path(args.out).expanduser().resolve() if args.out else artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", results)
    print(f"\nSaved: {out_dir / 'metrics.json'}")

    # Print summary table
    print("\n" + "=" * 70)
    print("Q3 — SUMMARIZATION RESULTS")
    print("=" * 70)
    print(f"{'Model':<12} {'R-1':>8} {'R-2':>8} {'R-L':>8} {'BLEU':>8} {'METEOR':>8} {'BERT-F1':>9}")
    print("-" * 70)
    for name, info in results["models"].items():
        m = info["metrics"]
        print(
            f"{name:<12} {m['ROUGE-1']:>8.2f} {m['ROUGE-2']:>8.2f} {m['ROUGE-L']:>8.2f} "
            f"{m['BLEU']:>8.2f} {m['METEOR']:>8.2f} {m['BERTScore-F1']:>9.2f}"
        )


if __name__ == "__main__":
    main()
