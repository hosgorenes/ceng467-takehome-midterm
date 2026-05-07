from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import evaluate as hf_eval
import pandas as pd
import torch
from bert_score import score as bert_score_fn

from q4_translation.config import Q4Config
from shared.io import read_json, write_json
from shared.paths import get_paths

warnings.filterwarnings("ignore")


def load_translations(path: Path) -> dict:
    data = read_json(path)
    return data


def compute_metrics(
    preds: list[str],
    refs: list[str],
    model_name: str,
    cfg: Q4Config,
    device: torch.device | str = "cpu",
) -> dict:
    print(f"\nComputing metrics for {model_name}...")

    bleu_metric = hf_eval.load("bleu")
    meteor_metric = hf_eval.load("meteor")
    chrf_metric = hf_eval.load("chrf")

    bleu_preds = [" ".join(p) if isinstance(p, list) else str(p) for p in preds]
    bleu_refs = [[" ".join(r)] if isinstance(r, list) else [str(r)] for r in refs]
    
    bleu_result = bleu_metric.compute(predictions=bleu_preds, references=bleu_refs)

    string_preds = [" ".join(p) if isinstance(p, list) else str(p) for p in preds]
    string_refs = [" ".join(r) if isinstance(r, list) else str(r) for r in refs]

    meteor_result = meteor_metric.compute(predictions=string_preds, references=string_refs)
    chrf_result = chrf_metric.compute(predictions=string_preds, references=string_refs)

    print("  Computing BERTScore...")
    subset = cfg.bertscore_subset
    P, R, F1 = bert_score_fn(
        string_preds[:subset],
        string_refs[:subset],
        lang="de",
        model_type="distilbert-base-multilingual-cased",
        verbose=False,
        device=device,
    )

    return {
        "model": model_name,
        "BLEU": round(bleu_result["bleu"] * 100, 2),
        "METEOR": round(meteor_result["meteor"] * 100, 2),
        "ChrF": round(chrf_result["score"], 2),
        "BERTScore-F1": round(F1.mean().item() * 100, 2),
    }


def error_analysis(
    preds: list[str],
    refs: list[str],
    sources: list[str],
    model_name: str,
) -> dict:
    string_preds = [" ".join(p) if isinstance(p, list) else str(p) for p in preds]
    
    short_outputs = [
        {"idx": i, "src": sources[i][:60], "pred": string_preds[i], "ref": refs[i][:50]}
        for i in range(len(string_preds))
        if len(string_preds[i].split()) < 3
    ]

    unk_outputs = [{"idx": i, "pred": string_preds[i][:80]} for i in range(len(string_preds)) if "<unk>" in string_preds[i]]

    long_idx = [i for i, s in enumerate(sources) if len(s.split()) > 20]

    return {
        "model": model_name,
        "short_outputs_count": len(short_outputs),
        "short_outputs_samples": short_outputs[:5],
        "unk_outputs_count": len(unk_outputs),
        "unk_outputs_samples": unk_outputs[:5],
        "long_sentence_count": len(long_idx),
    }


def qualitative_samples(
    seq_data: dict,
    marian_data: dict,
    indices: list[int] | None = None,
) -> list[dict]:
    if indices is None:
        indices = [0, 15, 99]

    samples = []
    for idx in indices:
        if idx < len(seq_data["sources"]) and idx < len(marian_data["sources"]):
            samples.append(
                {
                    "idx": idx,
                    "source_en": seq_data["sources"][idx],
                    "reference_de": seq_data["references"][idx],
                    "seq2seq_pred": " ".join(seq_data["predictions"][idx]) if isinstance(seq_data["predictions"][idx], list) else str(seq_data["predictions"][idx]),
                    "marian_pred": " ".join(marian_data["predictions"][idx]) if isinstance(marian_data["predictions"][idx], list) else str(marian_data["predictions"][idx]),
                }
            )
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory")
    ap.add_argument("--seq_dir", type=str, default="", help="Seq2Seq translations directory")
    ap.add_argument("--marian_dir", type=str, default="", help="MarianMT translations directory")
    args = ap.parse_args()

    cfg = Q4Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = get_paths().artifacts

    seq_path = Path(args.seq_dir).expanduser().resolve() if args.seq_dir else (artifacts / "q4" / "seq2seq")
    marian_path = Path(args.marian_dir).expanduser().resolve() if args.marian_dir else (artifacts / "q4" / "marian")
    out_dir = Path(args.out).expanduser().resolve() if args.out else (artifacts / "q4" / "evaluation")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading translations...")
    seq_data = load_translations(seq_path / "translations.json")
    marian_data = load_translations(marian_path / "translations.json")

    seq_metrics = compute_metrics(seq_data["predictions"], seq_data["references"], "Seq2Seq+Attention", cfg, device)
    marian_metrics = compute_metrics(marian_data["predictions"], marian_data["references"], "Helsinki-NLP", cfg, device)

    results_df = pd.DataFrame([seq_metrics, marian_metrics])
    print("\n" + "=" * 60)
    print("Q4 — MODEL COMPARISON (Multi30k Test Set)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    seq_errors = error_analysis(seq_data["predictions"], seq_data["references"], seq_data["sources"], "Seq2Seq+Attention")
    marian_errors = error_analysis(
        marian_data["predictions"], marian_data["references"], marian_data["sources"], "Helsinki-NLP"
    )

    qual_samples = qualitative_samples(seq_data, marian_data)

    print("\n" + "=" * 60)
    print("QUALITATIVE ANALYSIS — 3 EXAMPLES")
    print("=" * 60)
    for sample in qual_samples:
        print(f"\n{'─' * 58}")
        print(f"EXAMPLE #{sample['idx']}")
        print(f"  [Source EN]    : {sample['source_en']}")
        print(f"  [Reference DE] : {sample['reference_de']}")
        print(f"  [Seq2Seq]      : {sample['seq2seq_pred']}")
        print(f"  [Helsinki-NLP] : {sample['marian_pred']}")

    latex_table = r"""
\begin{table}[h]
\centering
\caption{Q4 Machine Translation -- Multi30k Test (EN$\rightarrow$DE)}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{BLEU} & \textbf{METEOR} & \textbf{ChrF} & \textbf{BERTScore} \\
\midrule
"""
    for row in [seq_metrics, marian_metrics]:
        latex_table += f"{row['model']} & {row['BLEU']} & {row['METEOR']} & {row['ChrF']} & {row['BERTScore-F1']} \\\\\n"

    latex_table += r"""\bottomrule
\end{tabular}
\label{tab:mt_results}
\end{table}"""

    print("\n\nLaTeX Table (copy to report):")
    print(latex_table)

    output = {
        "seq2seq_metrics": seq_metrics,
        "marian_metrics": marian_metrics,
        "seq2seq_errors": seq_errors,
        "marian_errors": marian_errors,
        "qualitative_samples": qual_samples,
        "latex_table": latex_table,
    }
    write_json(out_dir / "results.json", output)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()