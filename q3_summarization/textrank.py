from __future__ import annotations

import argparse
from pathlib import Path

from q3_summarization.config import Q3Config
from q3_summarization.preprocess import load_cnn_dailymail
from shared.io import write_json
from shared.paths import get_paths
from shared.seed import set_seed


def get_textrank_summarizer(language: str = "english"):
    from sumy.nlp.stemmers import Stemmer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.utils import get_stop_words

    stemmer = Stemmer(language)
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def textrank_summarize(text: str, n_sentences: int = 3, language: str = "english") -> str:
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser

    summarizer = get_textrank_summarizer(language)
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summary = summarizer(parser.document, n_sentences)
    return " ".join(str(s) for s in summary)


def generate_textrank_summaries(
    articles: list[str],
    n_sentences: int = 3,
    language: str = "english",
    verbose: bool = True,
) -> list[str]:
    summaries = []
    for i, art in enumerate(articles):
        summaries.append(textrank_summarize(art, n_sentences, language))
        if verbose and (i + 1) % 100 == 0:
            print(f"  TextRank: {i + 1}/{len(articles)} completed")
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory")
    ap.add_argument("--n-test", type=int, default=None)
    args = ap.parse_args()

    cfg = Q3Config(n_test=args.n_test or Q3Config.n_test)
    set_seed(cfg.seed)

    print("Loading CNN/DailyMail...")
    data = load_cnn_dailymail(cfg)

    print("Generating TextRank summaries...")
    summaries = generate_textrank_summaries(
        data.articles_clean,
        n_sentences=cfg.textrank_n_sentences,
        language=cfg.textrank_language,
    )

    out_dir = Path(args.out).expanduser().resolve() if args.out else (get_paths().artifacts / "q3" / "textrank")
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
