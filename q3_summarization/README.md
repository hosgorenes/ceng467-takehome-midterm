# Q3 — Text Summarization (CNN/DailyMail)

Bu klasör 2 özetleme yaklaşımını karşılaştırır:

- **TextRank (Extractive)**: Orijinal cümleleri seçerek özet oluşturur. Hallucination riski yok.
- **BART (Abstractive)**: Yeni cümleler üretir. Daha akıcı ama hallucination riski var.

## Kurulum

```bash
pip install -r requirements.txt
pip install sumy rouge-score bert-score sacrebleu evaluate
```

NLTK verileri (Colab'da ilk çalıştırmada):

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

## Çalıştırma

### 1) TextRank (Extractive) — GPU gerektirmez

```bash
python -m q3_summarization.textrank
```

### 2) BART (Abstractive) — GPU önerilir

```bash
python -m q3_summarization.bart
```

Hızlı test (daha az örnek):

```bash
python -m q3_summarization.textrank --n-test 50
python -m q3_summarization.bart --n-test 50
```

### 3) Metrikleri Hesapla

Her iki model çalıştıktan sonra:

```bash
python -m q3_summarization.evaluate
```

## Çıktılar

- `artifacts/q3/textrank/summaries.json`: TextRank özetleri + referanslar
- `artifacts/q3/bart/summaries.json`: BART özetleri + referanslar
- `artifacts/q3/metrics.json`: ROUGE/BLEU/METEOR/BERTScore + qualitative örnekler

## Metrikler

| Metrik | Açıklama |
|--------|----------|
| ROUGE-1/2/L | N-gram overlap (extractive-friendly) |
| BLEU | N-gram precision + brevity penalty |
| METEOR | Synonym/stem matching |
| BERTScore | Semantic similarity (abstractive-friendly) |
