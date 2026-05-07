from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from q4_translation.config import Q4Config
from q4_translation.preprocess import (
    EOS,
    SOS,
    UNK,
    build_vocabs,
    create_dataloaders,
    encode_sentence,
    load_multi30k,
    preprocess_text,
)
from shared.io import write_json
from shared.paths import get_paths
from shared.seed import set_seed


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(
            emb_dim,
            hid_dim,
            n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.drop(self.embedding(src))
        out, hid = self.rnn(emb)
        hid = torch.tanh(self.fc(torch.cat([hid[-2], hid[-1]], dim=1)))
        return out, hid


class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, enc_out):
        S = enc_out.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, S, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out], dim=2)))
        scores = self.v(energy).squeeze(2)
        return torch.softmax(scores, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attn = BahdanauAttention(hid_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim * 2, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt_tok, hidden, enc_out):
        tgt_tok = tgt_tok.unsqueeze(1)
        emb = self.drop(self.embedding(tgt_tok))
        a = self.attn(hidden, enc_out).unsqueeze(1)
        ctx = torch.bmm(a, enc_out)
        rnn_in = torch.cat([emb, ctx], dim=2)
        out, hid = self.rnn(rnn_in, hidden.unsqueeze(0))
        emb = emb.squeeze(1)
        out = out.squeeze(1)
        ctx = ctx.squeeze(1)
        pred = self.fc_out(torch.cat([out, ctx, emb], dim=1))
        return pred, hid.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, tgt_vocab_size: int, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_vocab_size = tgt_vocab_size
        self.device = device

    def forward(self, src, tgt, tf_ratio: float = 0.5):
        B, T = tgt.shape
        outputs = torch.zeros(B, T, self.tgt_vocab_size).to(self.device)
        enc_out, hidden = self.encoder(src)
        inp = tgt[:, 0]
        for t in range(1, T):
            pred, hidden = self.decoder(inp, hidden, enc_out)
            outputs[:, t] = pred
            teacher_force = random.random() < tf_ratio
            inp = tgt[:, t] if teacher_force else pred.argmax(1)
        return outputs


def greedy_decode(
    model: Seq2Seq,
    sentence: str,
    src_vocab: dict[str, int],
    tgt_vocab: dict[str, int],
    inv_tgt: dict[int, str],
    device: torch.device,
    max_len: int = 50,
) -> str:
    model.eval()
    tokens = encode_sentence(preprocess_text(sentence), src_vocab, max_len)
    src = torch.tensor(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        enc_out, hidden = model.encoder(src)
        inp = torch.tensor([tgt_vocab[SOS]]).to(device)
        result = []
        for _ in range(max_len):
            pred, hidden = model.decoder(inp, hidden, enc_out)
            top = pred.argmax(1)
            word = inv_tgt.get(top.item(), UNK)
            if word == EOS:
                break
            result.append(word)
            inp = top
    return " ".join(result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output directory")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = Q4Config(seq_epochs=args.epochs or Q4Config.seq_epochs)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Multi30k...")
    data = load_multi30k(cfg)
    vocabs = build_vocabs(data, cfg)
    train_loader, val_loader, test_loader = create_dataloaders(data, vocabs, cfg)

    print(f"Src vocab: {len(vocabs.src_vocab)}  Tgt vocab: {len(vocabs.tgt_vocab)}")

    encoder = Encoder(len(vocabs.src_vocab), cfg.seq_embed_dim, cfg.seq_hidden_dim, cfg.seq_num_layers, cfg.seq_dropout)
    decoder = Decoder(len(vocabs.tgt_vocab), cfg.seq_embed_dim, cfg.seq_hidden_dim, cfg.seq_dropout)
    model = Seq2Seq(encoder, decoder, len(vocabs.tgt_vocab), device).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Seq2Seq parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.seq_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    def train_epoch(tf_ratio: float):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(src, tgt, tf_ratio)
            out = out[:, 1:].reshape(-1, out.shape[-1])
            tgt2 = tgt[:, 1:].reshape(-1)
            loss = criterion(out, tgt2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.seq_grad_clip)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate_loss(loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for src, tgt in loader:
                src, tgt = src.to(device), tgt.to(device)
                out = model(src, tgt, tf_ratio=0)
                out = out[:, 1:].reshape(-1, out.shape[-1])
                tgt2 = tgt[:, 1:].reshape(-1)
                total_loss += criterion(out, tgt2).item()
        return total_loss / len(loader)

    print("\nSeq2Seq training started...")
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg.seq_epochs + 1):
        t0 = time.time()
        tr = train_epoch(cfg.seq_tf_ratio)
        val = evaluate_loss(val_loader)
        ppl = math.exp(min(val, 10))
        scheduler.step(val)
        print(f"Epoch {epoch:2d} | train_loss: {tr:.3f} | val_loss: {val:.3f} | val_ppl: {ppl:.2f} | {time.time() - t0:.0f}s")
        if val < best_val_loss:
            best_val_loss = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nGenerating test translations...")
    predictions = [
        greedy_decode(model, s, vocabs.src_vocab, vocabs.tgt_vocab, vocabs.inv_tgt, device) for s in data.test_src
    ]

    out_dir = Path(args.out).expanduser().resolve() if args.out else (get_paths().artifacts / "q4" / "seq2seq")
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": cfg.__dict__,
        "n_examples": len(predictions),
        "predictions": predictions,
        "references": data.test_tgt,
        "sources": data.test_src,
    }
    write_json(out_dir / "translations.json", output)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
