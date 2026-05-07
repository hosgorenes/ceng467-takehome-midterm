from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from q5_lm.config import Q5Config
from q5_lm.preprocess import UNK, encode_tokens, load_wikitext
from shared.io import write_json
from shared.paths import get_paths
from shared.seed import set_seed


class NGramModel:
    def __init__(self, n: int, alpha: float, vocab_size: int):
        self.n = n
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.ngram_counts: dict[tuple, Counter] = defaultdict(Counter)
        self.unigram_count: Counter = Counter()

    def train(self, token_ids: list[int]) -> None:
        self.unigram_count = Counter(token_ids)
        for i in range(len(token_ids) - self.n + 1):
            context = tuple(token_ids[i : i + self.n - 1])
            word = token_ids[i + self.n - 1]
            self.ngram_counts[context][word] += 1

    def prob(self, context: list[int], word: int) -> float:
        ctx = tuple(context[-(self.n - 1) :])
        count = self.ngram_counts[ctx][word] + self.alpha
        total = sum(self.ngram_counts[ctx].values()) + self.alpha * self.vocab_size
        return count / total

    def perplexity(self, token_ids: list[int]) -> float:
        log_prob = 0.0
        count = 0
        for i in range(self.n - 1, len(token_ids)):
            ctx = token_ids[i - (self.n - 1) : i]
            word = token_ids[i]
            p = self.prob(ctx, word)
            log_prob += math.log(max(p, 1e-10))
            count += 1
        return math.exp(-log_prob / count) if count > 0 else float("inf")

    def generate(
        self,
        seed_ids: list[int],
        inv_vocab: dict[int, str],
        n_words: int = 30,
        temperature: float = 1.0,
    ) -> str:
        result = list(seed_ids)
        for _ in range(n_words):
            ctx = result[-(self.n - 1) :]
            ctx_tup = tuple(ctx)
            cands = self.ngram_counts.get(ctx_tup, {})
            if not cands:
                words = list(self.unigram_count.keys())
                probs = np.array([self.unigram_count[w] for w in words], dtype=float)
                probs /= probs.sum()
                result.append(int(np.random.choice(words, p=probs)))
            else:
                words = list(cands.keys())
                counts = np.array([cands[w] for w in words], dtype=float)
                logits = np.log(counts + 1e-10) / temperature
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                result.append(int(np.random.choice(words, p=probs)))
        return " ".join(inv_vocab.get(i, UNK) for i in result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    cfg = Q5Config(
        ngram_n=args.n or Q5Config.ngram_n,
        ngram_alpha=args.alpha or Q5Config.ngram_alpha,
    )
    set_seed(cfg.seed)

    print("Loading WikiText-2...")
    data = load_wikitext(cfg)
    print(f"Vocab size: {len(data.vocab):,}")

    print(f"\nBuilding {cfg.ngram_n}-gram model...")
    model = NGramModel(cfg.ngram_n, cfg.ngram_alpha, len(data.vocab))
    model.train(data.train_ids)

    print("Computing perplexity...")
    val_ppl = model.perplexity(data.val_ids[: cfg.ngram_eval_subset])
    test_ppl = model.perplexity(data.test_ids[: cfg.ngram_eval_subset])

    print(f"  {cfg.ngram_n}-gram (Laplace α={cfg.ngram_alpha})")
    print(f"  Val PPL:  {val_ppl:.2f}")
    print(f"  Test PPL: {test_ppl:.2f}")

    seeds = ["the history of", "in the united", "science and technology"]
    samples = []
    print("\n--- N-gram generated samples ---")
    for seed in seeds:
        seed_ids = encode_tokens(seed.lower().split(), data.vocab)
        generated = model.generate(seed_ids, data.inv_vocab, n_words=cfg.gen_n_words, temperature=cfg.gen_temperature)
        print(f"Seed: '{seed}'")
        print(f"  → {generated}\n")
        samples.append({"seed": seed, "generated": generated})

    out_dir = Path(args.out).expanduser().resolve() if args.out else (get_paths().artifacts / "q5" / "ngram")
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "n": cfg.ngram_n,
            "alpha": cfg.ngram_alpha,
            "eval_subset": cfg.ngram_eval_subset,
        },
        "val_ppl": round(val_ppl, 2),
        "test_ppl": round(test_ppl, 2),
        "samples": samples,
    }
    write_json(out_dir / "results.json", output)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
