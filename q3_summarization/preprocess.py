from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

from q3_summarization.config import Q3Config
from shared.seed import set_seed


@dataclass(frozen=True)
class Q3Data:
    articles: list[str]
    articles_clean: list[str]
    references: list[str]


def clean_article(text: str) -> str:
    """Clean byline patterns in CNN/DailyMail."""
    text = re.sub(r"^\(CNN\)\s*--?\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_cnn_dailymail(cfg: Q3Config = Q3Config()) -> Q3Data:
    set_seed(cfg.seed)
    raw = load_dataset(cfg.dataset_name, cfg.dataset_version)
    test_data = raw["test"].select(range(cfg.n_test))

    articles = list(test_data["article"])
    references = list(test_data["highlights"])
    articles_clean = [clean_article(a) for a in articles]

    return Q3Data(
        articles=articles,
        articles_clean=articles_clean,
        references=references,
    )


def dataset_stats(data: Q3Data) -> dict[str, dict[str, float]]:
    art_lens = [len(a.split()) for a in data.articles_clean]
    ref_lens = [len(r.split()) for r in data.references]

    return {
        "articles": {
            "mean": float(np.mean(art_lens)),
            "median": float(np.median(art_lens)),
            "max": float(max(art_lens)) if art_lens else 0,
        },
        "references": {
            "mean": float(np.mean(ref_lens)),
            "median": float(np.median(ref_lens)),
            "max": float(max(ref_lens)) if ref_lens else 0,
        },
    }
