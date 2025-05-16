
# 👤 Author

**Harshal Gajjar**  
JHU CS | Section 601.666  
hgajjar1@jhu.edu

# Efficient Search Engine for Personal Notes & Documents

This project implements a modular, extensible search engine for personal `.txt` and `.md` files. It supports multiple retrieval models (TF-IDF, BM25, Dense Embeddings), WordNet-based query expansion, snippet generation, and IR evaluation using standard metrics.

---

## ✅ Features

- 🔍 **TF-IDF**, **BM25**, and **Dense (MiniLM)** retrieval models
- 🌱 **Query Expansion** using WordNet synonyms
- 📋 **Snippet Generation** for query-relevant document excerpts
- 🖥️ **Command-Line** and **Tkinter GUI**
- 📊 **Evaluation Suite** with P@5, nDCG@5, and MRR
- 📈 Visual comparison of IR models with/without query expansion

---

## 📁 Project Structure

```
project_root/
│
├── search_engine/
│   ├── file_loader.py
│   ├── retrieval_engine.py
│   ├── text_processor.py
│   ├── query_expander.py
│   ├── snippet_generator.py
│   ├── evaluator.py
│   └── eval_ground_truth.py
│
├── cli/
│   └── search_cli.py
│
├── data/                   
│
├── run_search.py
├── eval.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

If using query expansion, also run:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

### 2. Run the Search Engine (CLI)

```bash
python run_search.py
```

Follow the prompts to:
- Enter your query
- Choose method: `tfidf`, `bm25`, or `dense`
- Enable or disable query expansion
- View ranked documents and context snippets


### 3. Evaluate Retrieval Models

```bash
python eval.py
```

Evaluates:
- TF-IDF, BM25, and Dense
- With and without query expansion
- Using 15 manually annotated ground-truth queries

---

## 📊 Evaluation Sample Output

| Method | QE   | Precision@5 | nDCG@5 | MRR   |
|--------|------|-------------|--------|--------|
| TF-IDF | No   | 0.2533      | 0.2735 | 0.3911 |
| TF-IDF | Yes  | 0.2267      | 0.2728 | 0.4856 |
| BM25   | No   | 0.3733      | 0.4315 | 0.5889 |
| BM25   | Yes  | 0.4400      | 0.4990 | 0.6500 |
| Dense  | No   | 0.3067      | 0.3602 | 0.5356 |
| Dense  | Yes  | 0.1867      | 0.2495 | 0.4167 |

> BM25 + Query Expansion yielded the highest overall retrieval effectiveness.

---

## 📚 Dataset

The system uses a local collection of `.txt`/`.md` files organized in folders (e.g., `business/`, `sport/`, `politics/`, etc.).

A sample evaluation ground truth is defined in:

```
search_engine/eval_ground_truth.py
```

You may replace it or extend it with your own annotations.

---

## 💼 Requirements

- Python 3.8+
- `nltk`
- `scikit-learn`
- `rank-bm25`
- `sentence-transformers`
- `tkinter` (standard with Python)

Install them via:

```bash
pip install -r requirements.txt
```
