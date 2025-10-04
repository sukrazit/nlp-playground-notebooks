# NLP Playground — RNNs & Transformers (Notebooks)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-transformers-ffcc4d.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Hands-on NLP with two complementary notebooks:
- **01 — RNN/LSTM/GRU**: classic sequence models and training loops.
- **02 — Modern NLP Toolbox**: transformer pipelines (Hugging Face) + useful extras (TF-IDF / Word2Vec / FastText / embeddings, metrics, interpretation).

The goal is to show **clear, reproducible** experiments rather than a full-blown app.

---

## Contents

- `01_rnn_lstm_gru.ipynb` — tokenization → vocab → padded sequences → RNN/LSTM/GRU classifier; metrics (accuracy/F1/ROC-AUC), confusion matrix.  
- `02_modern_nlp_toolbox.ipynb` — HF `transformers` (AutoTokenizer/AutoModel*), `sentence-transformers`, TF-IDF baselines, visualizations, SHAP/LIME, UMAP.

> Models/figures are saved to `models/` and `artifacts/` (ignored by git).

---

## Quickstart

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**First run** (downloads small NLP resources):

```python
import nltk
nltk.download("punkt"); nltk.download("stopwords"); nltk.download("wordnet"); nltk.download("omw-1.4")
# Optional: spaCy small English model
# !python -m spacy download en_core_web_sm
```

Open notebooks in Jupyter / VS Code and run.

> **GPU (optional):** for CUDA wheels follow https://pytorch.org/get-started/ . On CPU everything runs too, just slower.

---

## Data

Put public/sample datasets under `data/`.  
Each notebook has a data cell showing expected paths or how to load via 🤗 `datasets`.

```
data/
  └─ (your files here)
models/        # saved weights (gitignored)
artifacts/     # figures/reports (gitignored)
```

---

## Reproducibility Tips
- Fix seeds (`random`, `numpy`, `torch`) where relevant.  
- Log key training params in the first markdown cell.  
- Keep outputs light in commits (show heads/top-k, not full dumps).

---

## Requirements

See `requirements.txt`. Main stacks:
- **Core:** numpy, pandas, scikit-learn, tqdm, scipy  
- **Viz:** matplotlib, seaborn, plotly, wordcloud  
- **NLP:** nltk, spacy, contractions, gensim  
- **Embeddings/Transformers:** transformers, datasets, accelerate, sentencepiece, sentence-transformers  
- **DL:** torch, torchtext  
- **Interpretability/Extras:** shap, lime, umap-learn, optuna, mlflow

---

## License

MIT — see [`LICENSE`](LICENSE).
