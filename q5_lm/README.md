# Q5 — Language Modeling (WikiText-2)

WikiText-2 dataset üzerinde dil modelleme.

## Setup (Colab)

```bash
!pip install datasets transformers torch evaluate nltk numpy pandas
```

## Çalıştırma

1. **N-gram Language Model** (Trigram + Laplace smoothing):
   ```bash
   !python -m q5_lm.ngram
   ```

2. **LSTM Language Model** (15 epoch):
   ```bash
   !python -m q5_lm.lstm_lm
   ```

3. **Evaluation** (her iki model için karşılaştırma):
   ```bash
   !python -m q5_lm.evaluate
   ```

## Çıktılar

```
artifacts/
└── q5/
    ├── ngram/
    │   └── results.json
    ├── lstm/
    │   └── results.json
    └── evaluation/
        └── results.json
```

## Metrikler

| Metrik       | Açıklama                                                    |
| ------------ | ----------------------------------------------------------- |
| Perplexity   | exp(cross-entropy loss) — düşük daha iyi                    |
| TTR          | Type-Token Ratio — lexical diversity ölçümü                 |

## Modüller

- `config.py` — Tüm hiperparametreler (seed, n-gram n, LSTM layers vb.)
- `preprocess.py` — WikiText-2 yükleme, tokenization, vocabulary
- `ngram.py` — N-gram dil modeli eğitimi + text generation
- `lstm_lm.py` — LSTM dil modeli eğitimi + text generation
- `evaluate.py` — Perplexity karşılaştırması + fluency metrics
