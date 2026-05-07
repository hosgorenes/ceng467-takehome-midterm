from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from q5_lm.config import Q5Config
from q5_lm.preprocess import UNK, batchify, encode_tokens, get_batch, load_wikitext
from shared.io import write_json
from shared.paths import get_paths
from shared.seed import set_seed


class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            n_layers,
            batch_first=False,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

        if tie_weights:
            assert embed_dim == hidden_dim, "embed_dim must equal hidden_dim for weight tying"
            self.fc.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, hidden=None):
        emb = self.drop(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        out = self.drop(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
            weight.new_zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
        )


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    return tuple(repackage_hidden(v) for v in h)


def generate_text(
    model: LSTMLanguageModel,
    seed_ids: list[int],
    inv_vocab: dict[int, str],
    vocab_size: int,
    device: torch.device,
    n_words: int = 40,
    temperature: float = 0.8,
) -> str:
    model.eval()
    inp = torch.tensor(seed_ids, dtype=torch.long).unsqueeze(1).to(device)
    hidden = model.init_hidden(1, device)

    result = list(seed_ids)
    with torch.no_grad():
        logits, hidden = model(inp, hidden)
        for _ in range(n_words):
            last_logit = logits[-1, 0] / temperature
            probs = torch.softmax(last_logit, dim=0).cpu().numpy()
            next_id = int(np.random.choice(vocab_size, p=probs))
            result.append(next_id)
            inp = torch.tensor([[next_id]], dtype=torch.long).to(device)
            logits, hidden = model(inp, hidden)

    return " ".join(inv_vocab.get(i, UNK) for i in result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = Q5Config(lstm_epochs=args.epochs or Q5Config.lstm_epochs)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading WikiText-2...")
    data = load_wikitext(cfg)
    vocab_size = len(data.vocab)
    print(f"Vocab size: {vocab_size:,}")

    train_data = batchify(data.train_ids, cfg.lstm_batch_size, device)
    val_data = batchify(data.val_ids, cfg.lstm_batch_size, device)
    test_data = batchify(data.test_ids, cfg.lstm_batch_size, device)

    print(f"LSTM data shapes: train={tuple(train_data.shape)} val={tuple(val_data.shape)} test={tuple(test_data.shape)}")

    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=cfg.lstm_hidden_dim,
        hidden_dim=cfg.lstm_hidden_dim,
        n_layers=cfg.lstm_num_layers,
        dropout=cfg.lstm_dropout,
        tie_weights=cfg.lstm_tie_weights,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LSTM LM parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lstm_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    def train_epoch():
        model.train()
        total_loss = 0
        hidden = model.init_hidden(cfg.lstm_batch_size, device)
        for i in range(0, train_data.size(0) - 1, cfg.lstm_seq_len):
            x, y = get_batch(train_data, i, cfg.lstm_seq_len)
            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()
            logits, hidden = model(x, hidden)
            loss = criterion(logits.view(-1, vocab_size), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.lstm_grad_clip)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / (train_data.size(0) // cfg.lstm_seq_len)

    def evaluate(data_source):
        model.eval()
        total_loss = 0
        n_batches = 0
        hidden = model.init_hidden(cfg.lstm_batch_size, device)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, cfg.lstm_seq_len):
                x, y = get_batch(data_source, i, cfg.lstm_seq_len)
                hidden = repackage_hidden(hidden)
                logits, hidden = model(x, hidden)
                total_loss += criterion(logits.view(-1, vocab_size), y).item()
                n_batches += 1
        return total_loss / n_batches

    print("\nLSTM LM training started...")
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg.lstm_epochs + 1):
        t0 = time.time()
        tr = train_epoch()
        val = evaluate(val_data)
        scheduler.step(val)
        ppl = math.exp(min(val, 10))
        print(f"Epoch {epoch:2d} | train_loss: {tr:.3f} | val_loss: {val:.3f} | val_ppl: {ppl:.2f} | {time.time() - t0:.0f}s")
        if val < best_val_loss:
            best_val_loss = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate(test_data)
    test_ppl = math.exp(min(test_loss, 10))
    val_ppl = math.exp(min(best_val_loss, 10))
    print(f"\nLSTM Test PPL: {test_ppl:.2f}")

    seeds = ["the history of", "in the united", "science and technology"]
    samples = []
    print("\n--- LSTM generated samples ---")
    for seed in seeds:
        seed_ids = encode_tokens(seed.lower().split(), data.vocab)
        generated = generate_text(
            model, seed_ids, data.inv_vocab, vocab_size, device, n_words=cfg.gen_n_words, temperature=cfg.gen_temperature
        )
        print(f"Seed: '{seed}'")
        print(f"  → {generated}\n")
        samples.append({"seed": seed, "generated": generated})

    out_dir = Path(args.out).expanduser().resolve() if args.out else (get_paths().artifacts / "q5" / "lstm")
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "embed_dim": cfg.lstm_embed_dim,
            "hidden_dim": cfg.lstm_hidden_dim,
            "num_layers": cfg.lstm_num_layers,
            "dropout": cfg.lstm_dropout,
            "tie_weights": cfg.lstm_tie_weights,
            "epochs": cfg.lstm_epochs,
        },
        "val_ppl": round(val_ppl, 2),
        "test_ppl": round(test_ppl, 2),
        "samples": samples,
    }
    write_json(out_dir / "results.json", output)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
