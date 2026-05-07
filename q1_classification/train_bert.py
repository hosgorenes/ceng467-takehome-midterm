from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from q1_classification.config import Q1Config
from q1_classification.metrics import compute_metrics, misclassified_examples, save_confusion_matrix_png, save_run_summary
from q1_classification.preprocess import load_imdb_splits
from shared.paths import get_paths
from shared.seed import set_seed


@dataclass(frozen=True)
class BertTrainConfig:
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 16
    warmup_ratio: float = 0.1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory (defaults to artifacts/q1/bert)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from torch.optim import AdamW

    cfg = Q1Config()
    train_cfg = BertTrainConfig(epochs=args.epochs or BertTrainConfig.epochs)
    model_name = args.model or cfg.bert_model

    set_seed(cfg.seed)
    splits = load_imdb_splits(cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class IMDbBertDataset(Dataset):
        def __init__(self, texts: list[str], labels: np.ndarray):
            self.enc = tokenizer(
                texts,
                max_length=cfg.bert_max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int):
            return {
                "input_ids": self.enc["input_ids"][idx],
                "attention_mask": self.enc["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    train_loader = DataLoader(
        IMDbBertDataset(splits.train["clean"].tolist(), splits.train["label"].to_numpy()),
        batch_size=train_cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        IMDbBertDataset(splits.val["clean"].tolist(), splits.val["label"].to_numpy()),
        batch_size=train_cfg.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        IMDbBertDataset(splits.test["clean"].tolist(), splits.test["label"].to_numpy()),
        batch_size=train_cfg.batch_size,
        shuffle=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    opt = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=0.01)

    total_steps = len(train_loader) * train_cfg.epochs
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    def run_eval(loader):
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                pred = logits.argmax(1)
                all_p.extend(pred.detach().cpu().numpy().tolist())
                all_y.extend(batch["labels"].detach().cpu().numpy().tolist())
        return np.array(all_y), np.array(all_p)

    best_val_f1 = -1.0
    best_state = None
    model.train()
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()

        yv, pv = run_eval(val_loader)
        m = compute_metrics(yv, pv)
        print(f"Epoch {epoch}/{train_cfg.epochs} | val_acc={m.accuracy:.4f} val_f1={m.macro_f1:.4f}")
        if m.macro_f1 > best_val_f1:
            best_val_f1 = m.macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    yt, pt = run_eval(test_loader)
    m_test = compute_metrics(yt, pt)

    root = get_paths().artifacts / "q1" / "bert"
    out_dir = Path(args.out).expanduser().resolve() if args.out else root

    save_confusion_matrix_png(yt, pt, out_dir / "confusion.png", title=f"Q1 {model_name}")
    summary = {
        "config": cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "model": model_name,
        "test": m_test.__dict__,
        "errors": misclassified_examples(splits.test["text"].tolist(), yt, pt, k=5),
    }
    save_run_summary(out_dir, summary)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()

