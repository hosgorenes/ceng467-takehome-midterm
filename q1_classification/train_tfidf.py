from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from q1_classification.config import Q1Config
from q1_classification.metrics import compute_metrics, misclassified_examples, save_confusion_matrix_png, save_run_summary
from q1_classification.preprocess import load_imdb_splits
from shared.paths import get_paths
from shared.seed import set_seed


def build_vectorizers(cfg: Q1Config) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    uni = TfidfVectorizer(
        max_features=cfg.tfidf_max_features_uni,
        ngram_range=(1, 1),
        min_df=cfg.tfidf_min_df_uni,
        sublinear_tf=True,
    )
    bi = TfidfVectorizer(
        max_features=cfg.tfidf_max_features_bi,
        ngram_range=(1, 2),
        min_df=cfg.tfidf_min_df_bi,
        sublinear_tf=True,
    )
    return uni, bi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory (defaults to artifacts/q1/tfidf)")
    args = ap.parse_args()

    cfg = Q1Config()
    set_seed(cfg.seed)
    splits = load_imdb_splits(cfg)

    uni_vec, bi_vec = build_vectorizers(cfg)

    X_train_uni = uni_vec.fit_transform(splits.train["clean"])
    X_val_uni = uni_vec.transform(splits.val["clean"])
    X_test_uni = uni_vec.transform(splits.test["clean"])

    X_train_bi = bi_vec.fit_transform(splits.train["clean"])
    X_val_bi = bi_vec.transform(splits.val["clean"])
    X_test_bi = bi_vec.transform(splits.test["clean"])

    y_train = splits.train["label"].to_numpy()
    y_val = splits.val["label"].to_numpy()
    y_test = splits.test["label"].to_numpy()

    models = {
        "lr_unigram": (LogisticRegression(max_iter=1000, C=1.0, random_state=cfg.seed), X_train_uni, X_val_uni, X_test_uni),
        "lr_bigram": (LogisticRegression(max_iter=1000, C=1.0, random_state=cfg.seed), X_train_bi, X_val_bi, X_test_bi),
        "svm_unigram": (LinearSVC(C=1.0, max_iter=2000, random_state=cfg.seed), X_train_uni, X_val_uni, X_test_uni),
        "svm_bigram": (LinearSVC(C=1.0, max_iter=2000, random_state=cfg.seed), X_train_bi, X_val_bi, X_test_bi),
    }

    root = get_paths().artifacts / "q1" / "tfidf"
    out_dir = Path(args.out).expanduser().resolve() if args.out else root

    summary: dict[str, object] = {"config": cfg.__dict__, "models": {}}

    for name, (model, Xtr, Xva, Xte) in models.items():
        model.fit(Xtr, y_train)
        pred_val = model.predict(Xva)
        pred_test = model.predict(Xte)

        m_val = compute_metrics(y_val, pred_val)
        m_test = compute_metrics(y_test, pred_test)

        model_dir = out_dir / name
        save_confusion_matrix_png(y_test, pred_test, model_dir / "confusion.png", title=f"Q1 {name}")
        (summary["models"]).__setitem__(
            name,
            {
                "val": m_val.__dict__,
                "test": m_test.__dict__,
                "errors": misclassified_examples(
                    texts=splits.test["text"].tolist(),
                    y_true=y_test,
                    y_pred=pred_test,
                    k=5,
                ),
            },
        )

    save_run_summary(out_dir, summary)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()

