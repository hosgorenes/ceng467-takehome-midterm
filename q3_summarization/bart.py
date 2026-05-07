from __future__ import annotations

import argparse
from pathlib import Path

import torch

from q3_summarization.config import Q3Config
from q3_summarization.preprocess import load_cnn_dailymail
from shared.io import write_json
from shared.paths import get_paths
from shared.seed import set_seed


def truncate_words(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])


def generate_bart_summaries(
    articles: list[str],
    cfg: Q3Config = Q3Config(),
    verbose: bool = True,
) -> list[str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading BART model: {cfg.bart_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.bart_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.bart_model, torch_dtype=dtype).to(device)
    model.eval()

    articles_trunc = [truncate_words(a, cfg.bart_max_input_words) for a in articles]
    summaries = []

    n = len(articles_trunc)
    bs = cfg.bart_batch_size

    print(f"Generating BART summaries (batch_size={bs})...")
    for i in range(0, n, bs):
        batch = articles_trunc[i : i + bs]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=cfg.bart_max_output_length,
                min_length=cfg.bart_min_output_length,
                length_penalty=2.0,
                num_beams=cfg.bart_num_beams,
                early_stopping=True,
            )
        batch_summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        summaries.extend(batch_summaries)
        if verbose and (i + bs) % 100 == 0:
            print(f"  BART: {min(i + bs, n)}/{n} completed")

    return summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory")
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    cfg = Q3Config(
        n_test=args.n_test or Q3Config.n_test,
        bart_model=args.model or Q3Config.bart_model,
    )
    set_seed(cfg.seed)

    print("Loading CNN/DailyMail...")
    data = load_cnn_dailymail(cfg)

    summaries = generate_bart_summaries(data.articles_clean, cfg)

    out_dir = Path(args.out).expanduser().resolve() if args.out else (get_paths().artifacts / "q3" / "bart")
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": cfg.__dict__,
        "n_examples": len(summaries),
        "summaries": summaries,
        "references": data.references,
    }
    write_json(out_dir / "summaries.json", output)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
