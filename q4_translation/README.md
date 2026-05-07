# Q4 — Machine Translation (EN → DE)

Multi30k dataset üzerinde EN→DE çeviri.

## Setup (Colab)

```bash
!pip install datasets transformers torch sacrebleu evaluate bert-score sentencepiece
```

## Çalıştırma

1. **Seq2Seq + Bahdanau Attention** (15 epoch):
   ```bash
   !python -m q4_translation.seq2seq
   ```

2. **MarianMT (Helsinki-NLP)** (pretrained transformer):
   ```bash
   !python -m q4_translation.marian
   ```

3. **Evaluation** (her iki model için metrikleri hesapla):
   ```bash
   !python -m q4_translation.evaluate
   ```

## Çıktılar

```
artifacts/
└── q4/
    ├── seq2seq/
    │   └── translations.json
    ├── marian/
    │   └── translations.json
    └── evaluation/
        └── results.json
```

## Metrikler

| Metrik       | Açıklama                                                    |
| ------------ | ----------------------------------------------------------- |
| BLEU         | N-gram precision + brevity penalty                          |
| METEOR       | Stemming + synonyms ile alignment                           |
| ChrF         | Character n-gram F-score (morfololik diller için uygun)     |
| BERTScore    | Contextual embedding bazlı semantik benzerlik               |

## Modüller

- `config.py` — Tüm hiperparametreler (seed, model, batch size vb.)
- `preprocess.py` — Multi30k yükleme, preprocessing, vocabulary, dataloader
- `seq2seq.py` — Encoder-Decoder + Bahdanau Attention eğitimi
- `marian.py` — Helsinki-NLP MarianMT ile pretrained inference
- `evaluate.py` — BLEU/METEOR/ChrF/BERTScore hesaplama + error analysis
