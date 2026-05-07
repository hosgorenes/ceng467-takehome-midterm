# Q2 — Named Entity Recognition (CoNLL-2003)

Bu klasör 2 yaklaşımı karşılaştırır:

- **BiLSTM‑CRF**: word‑level BiLSTM + CRF decode (BIO geçişlerini daha tutarlı yapar)
- **BERT token classification**: `bert-base-cased` fine‑tuning + subword alignment

## Kurulum

```bash
pip install -r requirements.txt
```

Ek paketler:

- `seqeval`
- `torchcrf` (CRF katmanı için)

## Çalıştırma

### BiLSTM‑CRF

```bash
python -m q2_ner.train_bilstm_crf
```

### BERT NER

```bash
python -m q2_ner.train_bert_ner
```

## Çıktılar

Her model varsayılan olarak şu dosyayı üretir:

- `artifacts/q2/<model>/metrics.json`: Precision/Recall/F1 + örnek boundary/confusion hataları
