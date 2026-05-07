# Q1 — Text Classification (IMDb)

Bu klasör, **IMDb sentiment classification** için 3 temsil türünü karşılaştırır:

- **Sparse**: TF‑IDF + Logistic Regression / Linear SVM
- **Dense**: BiLSTM (scratch)
- **Contextual**: DistilBERT fine‑tuning

## Kurulum

Kök dizinde:

```bash
pip install -r requirements.txt
```

## Çalıştırma

### TF‑IDF (LR/SVM)

```bash
python -m q1_classification.train_tfidf
```

Çıktılar varsayılan olarak `artifacts/q1/tfidf/` altında oluşur (`metrics.json`, `confusion.png` vb.).

### BiLSTM

```bash
python -m q1_classification.train_bilstm
```

### DistilBERT

```bash
python -m q1_classification.train_bert
```

Model değiştirmek için:

```bash
python -m q1_classification.train_bert --model distilbert-base-uncased
```

## Üretilen Artefact’lar

- `artifacts/q1/**/metrics.json`: val/test metrikleri + 5 hata örneği
- `artifacts/q1/**/confusion.png`: confusion matrix

