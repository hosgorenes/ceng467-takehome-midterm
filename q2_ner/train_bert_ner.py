from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

from q2_ner.config import Q2Config
from q2_ner.metrics import analyze_boundary_and_confusion_errors, compute_ner_metrics
from q2_ner.preprocess import load_conll, tokenize_and_align_labels
from shared.io import write_json
from shared.paths import get_paths
from shared.seed import set_seed


@dataclass(frozen=True)
class BertNERTrainConfig:
    batch_size: int = 32
    epochs: int = 4
    lr: float = 3e-5
    warmup_ratio: float = 0.1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory (defaults to artifacts/q2/bert)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from seqeval.metrics import classification_report as seq_report

    cfg = Q2Config()
    set_seed(cfg.seed)
    raw, ls = load_conll(cfg)

    model_name = args.model or cfg.bert_model
    tokenized, _tok = tokenize_and_align_labels(raw, tokenizer_name=model_name, max_len=cfg.max_len)

    train_cfg = BertNERTrainConfig(epochs=args.epochs or BertNERTrainConfig.epochs)

    train_loader = DataLoader(tokenized["train"], batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(tokenized["validation"], batch_size=train_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(tokenized["test"], batch_size=train_cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(ls.labels),
        id2label=ls.id2label,
        label2id=ls.label2id,
    ).to(device)

    opt = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=0.01)
    total_steps = len(train_loader) * train_cfg.epochs
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    def predict(loader):
        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labs = batch["labels"].to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                preds = logits.argmax(-1).detach().cpu().tolist()
                labs = labs.detach().cpu().tolist()

                for p_seq, t_seq in zip(preds, labs):
                    p_list, t_list = [], []
                    for p, t in zip(p_seq, t_seq):
                        if t == -100:
                            continue
                        p_list.append(ls.id2label[int(p)])
                        t_list.append(ls.id2label[int(t)])
                    all_true.append(t_list)
                    all_pred.append(p_list)
        return all_true, all_pred

    best_val_f1 = -1.0
    best_state = None
    model.train()
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()

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

    out_dir = Path(args.out).expanduser().resolve() if args.out else (get_paths().artifacts / "q2" / "bert")
    out_dir.mkdir(parents=True, exist_ok=True)

    token_seqs = raw["test"]["tokens"]
    error_examples = analyze_boundary_and_confusion_errors(t_true, t_pred, token_seqs, limit_each=5)

    summary = {
        "config": cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "model": model_name,
        "test": m_test.__dict__,
        "error_examples": error_examples,
        "classification_report": seq_report(t_true, t_pred, digits=4),
    }
    write_json(out_dir / "metrics.json", summary)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()

