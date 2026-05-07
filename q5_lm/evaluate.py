from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from q5_lm.config import Q5Config
from shared.io import read_json, write_json
from shared.paths import get_paths


def avg_sent_len(texts: list[str]) -> float:
    lens = []
    for t in texts:
        sents = re.split(r"[.!?]", t)
        lens += [len(s.split()) for s in sents if s.strip()]
    return round(np.mean(lens), 2) if lens else 0


def type_token_ratio(texts: list[str]) -> float:
    all_tokens = " ".join(texts).split()
    return round(len(set(all_tokens)) / max(len(all_tokens), 1), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngram_dir", type=str, default="", help="N-gram results directory")
    ap.add_argument("--lstm_dir", type=str, default="", help="LSTM results directory")
    ap.add_argument("--out", type=str, default="", help="Output directory")
    args = ap.parse_args()

    cfg = Q5Config()
    artifacts = get_paths().artifacts / "q5"

    ngram_path = Path(args.ngram_dir) if args.ngram_dir else artifacts / "ngram"
    lstm_path = Path(args.lstm_dir) if args.lstm_dir else artifacts / "lstm"
    out_dir = Path(args.out).expanduser().resolve() if args.out else artifacts / "evaluation"

    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"models": []}

    ngram_data = None
    lstm_data = None

    if (ngram_path / "results.json").exists():
        ngram_data = read_json(ngram_path / "results.json")
        ngram_cfg = ngram_data.get("config", {})
        results["models"].append(
            {
                "model": f"{ngram_cfg.get('n', 3)}-gram (Laplace α={ngram_cfg.get('alpha', 0.1)})",
                "val_ppl": ngram_data.get("val_ppl"),
                "test_ppl": ngram_data.get("test_ppl"),
                "notes": "No GPU, fast",
            }
        )
    else:
        print(f"Skipping N-gram: {ngram_path / 'results.json'} not found")

    if (lstm_path / "results.json").exists():
        lstm_data = read_json(lstm_path / "results.json")
        lstm_cfg = lstm_data.get("config", {})
        results["models"].append(
            {
                "model": f"LSTM ({lstm_cfg.get('num_layers', 2)}L, hid={lstm_cfg.get('hidden_dim', 512)}, tie_weights={lstm_cfg.get('tie_weights', True)})",
                "val_ppl": lstm_data.get("val_ppl"),
                "test_ppl": lstm_data.get("test_ppl"),
                "notes": "Weight tying, BPTT",
            }
        )
    else:
        print(f"Skipping LSTM: {lstm_path / 'results.json'} not found")

    if not results["models"]:
        print("No results found. Run ngram.py and lstm_lm.py first.")
        return

    df = pd.DataFrame(results["models"])
    print("=" * 60)
    print("Q5 — LANGUAGE MODEL COMPARISON (WikiText-2)")
    print("=" * 60)
    print(df.to_string(index=False))

    if ngram_data and lstm_data:
        ngram_samples = [s["generated"] for s in ngram_data.get("samples", [])]
        lstm_samples = [s["generated"] for s in lstm_data.get("samples", [])]

        if ngram_samples and lstm_samples:
            print("\n" + "=" * 60)
            print("GENERATION COMPARISON")
            print("=" * 60)
            for ns, ls in zip(ngram_data.get("samples", []), lstm_data.get("samples", [])):
                print(f"\nSeed: '{ns['seed']}'")
                print(f"  [N-gram] → {ns['generated'][:100]}...")
                print(f"  [LSTM  ] → {ls['generated'][:100]}...")

            print("\n" + "=" * 60)
            print("FLUENCY PROXY METRICS")
            print("=" * 60)
            print(f"{'Metric':<25} {'N-gram':>10} {'LSTM':>10}")
            print("-" * 48)
            print(f"{'Avg sentence length':<25} {avg_sent_len(ngram_samples):>10.2f} {avg_sent_len(lstm_samples):>10.2f}")
            print(f"{'Type-Token Ratio (TTR)':<25} {type_token_ratio(ngram_samples):>10.4f} {type_token_ratio(lstm_samples):>10.4f}")
            print(f"{'Test Perplexity':<25} {ngram_data['test_ppl']:>10.2f} {lstm_data['test_ppl']:>10.2f}")

            results["fluency"] = {
                "ngram": {
                    "avg_sent_len": avg_sent_len(ngram_samples),
                    "ttr": type_token_ratio(ngram_samples),
                },
                "lstm": {
                    "avg_sent_len": avg_sent_len(lstm_samples),
                    "ttr": type_token_ratio(lstm_samples),
                },
            }

    latex_table = r"""
\begin{table}[h]
\centering
\caption{Q5 Language Model Comparison -- WikiText-2 Test Set}
\begin{tabular}{lccp{4.5cm}}
\toprule
\textbf{Model} & \textbf{Val PPL} & \textbf{Test PPL} & \textbf{Notes} \\
\midrule
"""
    for row in results["models"]:
        latex_table += f"{row['model']} & {row['val_ppl']} & {row['test_ppl']} & {row['notes']} \\\\\n"

    latex_table += r"""\bottomrule
\end{tabular}
\label{tab:lm_results}
\end{table}"""

    print("\n\nLaTeX Table (copy to report):")
    print(latex_table)

    results["latex_table"] = latex_table

    write_json(out_dir / "results.json", results)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
