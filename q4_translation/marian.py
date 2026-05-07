from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

from q4_translation.config import Q4Config
from q4_translation.preprocess import load_multi30k
from shared.io import write_json
from shared.paths import get_paths
from shared.seed import set_seed


def translate_batch(
    sentences: list[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: torch.device,
    batch_size: int = 32,
    verbose: bool = True,
) -> list[str]:
    translations = []
    n = len(sentences)

    for i in range(0, n, batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, num_beams=4, max_length=128, early_stopping=True)
        translations.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        if verbose and (i + batch_size) % 200 == 0:
            print(f"  MarianMT: {min(i + batch_size, n)}/{n} translated")

    return translations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory")
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    cfg = Q4Config(marian_model=args.model or Q4Config.marian_model)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Multi30k...")
    data = load_multi30k(cfg)

    print(f"Loading MarianMT: {cfg.marian_model}")
    tokenizer = MarianTokenizer.from_pretrained(cfg.marian_model)
    model = MarianMTModel.from_pretrained(cfg.marian_model).to(device)
    model.eval()

    print("\nGenerating MarianMT translations...")
    predictions = translate_batch(
        data.test_src_raw,
        tokenizer,
        model,
        device,
        batch_size=cfg.marian_batch_size,
    )

    out_dir = Path(args.out).expanduser().resolve() if args.out else (get_paths().artifacts / "q4" / "marian")
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": cfg.__dict__,
        "model": cfg.marian_model,
        "n_examples": len(predictions),
        "predictions": predictions,
        "references": data.test_tgt_raw,
        "sources": data.test_src_raw,
    }
    write_json(out_dir / "translations.json", output)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
