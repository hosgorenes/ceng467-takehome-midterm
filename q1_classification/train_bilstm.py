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
class BiLSTMTrainConfig:
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    epochs: int = 6
    batch_size: int = 64
    grad_clip: float = 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory (defaults to artifacts/q1/bilstm)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader, Dataset
    from torch.nn.utils.rnn import pad_sequence

    cfg = Q1Config()
    set_seed(cfg.seed)
    splits = load_imdb_splits(cfg)

    train_cfg = BiLSTMTrainConfig(epochs=args.epochs or BiLSTMTrainConfig.epochs)

    PAD_IDX = 0
    UNK_IDX = 1

    # --- vocab ---
    from collections import Counter

    counter = Counter()
    for text in splits.train["clean"].tolist():
        counter.update(text.split())
    vocab: dict[str, int] = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for w, _ in counter.most_common(cfg.bilstm_max_vocab - 2):
        vocab[w] = len(vocab)

    def encode(text: str) -> list[int]:
        toks = text.split()[: cfg.bilstm_max_len]
        return [vocab.get(t, UNK_IDX) for t in toks]

    class SeqDataset(Dataset):
        def __init__(self, texts: list[str], labels: np.ndarray):
            self.x = [torch.tensor(encode(t), dtype=torch.long) for t in texts]
            self.y = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int):
            return self.x[idx], self.y[idx]

    def collate(batch):
        seqs, labels = zip(*batch)
        padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
        return padded, torch.stack(labels)

    train_loader = DataLoader(
        SeqDataset(splits.train["clean"].tolist(), splits.train["label"].to_numpy()),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        SeqDataset(splits.val["clean"].tolist(), splits.val["label"].to_numpy()),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        SeqDataset(splits.test["clean"].tolist(), splits.test["label"].to_numpy()),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class BiLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(vocab), train_cfg.embed_dim, padding_idx=PAD_IDX)
            self.lstm = nn.LSTM(
                input_size=train_cfg.embed_dim,
                hidden_size=train_cfg.hidden_dim,
                num_layers=train_cfg.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=train_cfg.dropout if train_cfg.num_layers > 1 else 0.0,
            )
            self.drop = nn.Dropout(train_cfg.dropout)
            self.fc = nn.Linear(train_cfg.hidden_dim * 2, 2)

        def forward(self, x):
            emb = self.drop(self.embedding(x))
            _, (h, _) = self.lstm(emb)
            h = torch.cat([h[-2], h[-1]], dim=1)
            return self.fc(self.drop(h))

    model = BiLSTMClassifier().to(device)
    opt = Adam(model.parameters(), lr=train_cfg.lr)
    crit = nn.CrossEntropyLoss()

    def run_eval(loader):
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                p = model(x).argmax(1)
                all_p.extend(p.detach().cpu().numpy().tolist())
                all_y.extend(y.detach().cpu().numpy().tolist())
        return np.array(all_y), np.array(all_p)

    best_val_f1 = -1.0
    best_state = None
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()

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

    root = get_paths().artifacts / "q1" / "bilstm"
    out_dir = Path(args.out).expanduser().resolve() if args.out else root

    save_confusion_matrix_png(yt, pt, out_dir / "confusion.png", title="Q1 BiLSTM")
    summary = {
        "config": cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "test": m_test.__dict__,
        "errors": misclassified_examples(splits.test["text"].tolist(), yt, pt, k=5),
    }
    save_run_summary(out_dir, summary)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()

