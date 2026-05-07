from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from q2_ner.config import Q2Config
from q2_ner.metrics import analyze_boundary_and_confusion_errors, compute_ner_metrics
from q2_ner.preprocess import load_conll
from shared.io import write_json
from shared.paths import get_paths
from shared.seed import set_seed


@dataclass(frozen=True)
class BiLSTMCRFTrainConfig:
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.33
    lr: float = 1e-3
    epochs: int = 8
    batch_size: int = 64
    max_len: int = 128
    grad_clip: float = 5.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory (defaults to artifacts/q2/bilstm_crf)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchcrf import CRF
    from seqeval.metrics import classification_report as seq_report
    from collections import Counter

    cfg = Q2Config()
    set_seed(cfg.seed)
    raw, ls = load_conll(cfg)

    train_cfg = BiLSTMCRFTrainConfig(epochs=args.epochs or BiLSTMCRFTrainConfig.epochs, max_len=cfg.max_len)

    PAD_IDX = 0
    UNK_IDX = 1
    IGNORE_IDX = -100

    counter = Counter()
    for toks in raw["train"]["tokens"]:
        counter.update(toks)
    vocab: dict[str, int] = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for w, _ in counter.most_common(20_000 - 2):
        vocab[w] = len(vocab)

    def encode_tokens(tokens: list[str]) -> list[int]:
        ids = [vocab.get(t, UNK_IDX) for t in tokens[: train_cfg.max_len]]
        ids += [PAD_IDX] * max(0, train_cfg.max_len - len(ids))
        return ids

    def encode_labels(tags: list[int]) -> list[int]:
        labs = list(tags[: train_cfg.max_len])
        labs += [IGNORE_IDX] * max(0, train_cfg.max_len - len(labs))
        return labs

    class ConllWordDataset(Dataset):
        def __init__(self, split: str):
            self.ds = raw[split]

        def __len__(self) -> int:
            return len(self.ds)

        def __getitem__(self, idx: int):
            tokens = self.ds[idx]["tokens"]
            tags = self.ds[idx]["ner_tags"]
            x = torch.tensor(encode_tokens(tokens), dtype=torch.long)
            y = torch.tensor(encode_labels(tags), dtype=torch.long)
            return x, y

    train_loader = DataLoader(ConllWordDataset("train"), batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ConllWordDataset("validation"), batch_size=train_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(ConllWordDataset("test"), batch_size=train_cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class BiLSTMCRF(nn.Module):
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
            self.proj = nn.Linear(train_cfg.hidden_dim * 2, len(ls.labels))
            self.crf = CRF(len(ls.labels), batch_first=True)

        def emissions(self, x):
            emb = self.drop(self.embedding(x))
            out, _ = self.lstm(emb)
            return self.proj(self.drop(out))

        def loss(self, x, tags, mask):
            emit = self.emissions(x)
            return -self.crf(emit, tags, mask=mask, reduction="mean")

        def decode(self, x, mask):
            emit = self.emissions(x)
            return self.crf.decode(emit, mask=mask)

    model = BiLSTMCRF().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    def predict(loader):
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                mask = (y != IGNORE_IDX)
                safe_y = y.clone()
                safe_y[~mask] = 0
                pred_ids = model.decode(x, mask=mask)
                safe_y = safe_y.detach().cpu().tolist()
                mask = mask.detach().cpu().tolist()
                for p_seq, t_seq, m_seq in zip(pred_ids, safe_y, mask):
                    t_labels = [ls.id2label[int(t)] for t, m in zip(t_seq, m_seq) if m]
                    p_labels = [ls.id2label[int(p)] for p in p_seq]
                    min_len = min(len(t_labels), len(p_labels))
                    all_true.append(t_labels[:min_len])
                    all_pred.append(p_labels[:min_len])
        return all_true, all_pred

    best_val_f1 = -1.0
    best_state = None
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = (y != IGNORE_IDX)
            safe_y = y.clone()
            safe_y[~mask] = 0
            opt.zero_grad()
            loss = model.loss(x, safe_y, mask=mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()

        v_true, v_pred = predict(val_loader)
        m = compute_ner_metrics(v_true, v_pred)
        print(f"Epoch {epoch}/{train_cfg.epochs} | val_f1={m.f1:.4f}")
        if m.f1 > best_val_f1:
            best_val_f1 = m.f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    t_true, t_pred = predict(test_loader)
    m_test = compute_ner_metrics(t_true, t_pred)

    out_dir = Path(args.out).expanduser().resolve() if args.out else (get_paths().artifacts / "q2" / "bilstm_crf")
    out_dir.mkdir(parents=True, exist_ok=True)

    token_seqs = raw["test"]["tokens"]
    error_examples = analyze_boundary_and_confusion_errors(t_true, t_pred, token_seqs, limit_each=5)

    summary = {
        "config": cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "test": m_test.__dict__,
        "error_examples": error_examples,
        "classification_report": seq_report(t_true, t_pred, digits=4),
    }
    write_json(out_dir / "metrics.json", summary)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()

