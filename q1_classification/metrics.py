from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from shared.io import ensure_dir, write_json


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float


def compute_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
    )


def save_confusion_matrix_png(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    out_path: Path,
    title: str,
    labels: tuple[str, str] = ("Neg", "Pos"),
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    ensure_dir(out_path.parent)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(labels), yticklabels=list(labels))
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_run_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    ensure_dir(out_dir)
    write_json(out_dir / "metrics.json", summary)


def misclassified_examples(
    texts: list[str],
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    k: int = 5,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if int(t) != int(p):
            out.append(
                {
                    "idx": i,
                    "true": int(t),
                    "pred": int(p),
                    "text_excerpt": texts[i][:300],
                }
            )
            if len(out) >= k:
                break
    return out

