# 📚 Conversational Semantic Search for Electronic Theses and Dissertations (ETDs)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

## Overview

This repository presents an AI-powered semantic retrieval framework for Electronic Theses and Dissertations (ETDs). The system enables users to search scholarly repositories using natural language instead of relying solely on keyword matching.

The framework combines dense sentence embeddings, vector similarity search, explainable retrieval, grounded summaries, metadata-aware filtering, and interactive analytics to improve the discovery of academic documents.

This work accompanies the research paper:

> **Conversational Semantic Search for Electronic Theses and Dissertations Using Dense Retrieval and Explainable AI**

Submitted to **ETD 2026 – IIT Delhi**

---

## Features

- Semantic search using Sentence Transformers
- FAISS vector indexing
- Natural language querying
- Explainable retrieval
- Grounded retrieval summaries
- Metadata-aware filtering
- Interactive analytics dashboard
- Information Retrieval evaluation
- BM25 keyword baseline comparison
- Streamlit web application

---

## System Architecture

```
ETD Metadata
      │
Metadata Processing
      │
Sentence Transformer
      │
Dense Embeddings
      │
FAISS Vector Index
      │
Natural Language Query
      │
Semantic Retrieval
      │
Metadata Filtering
      │
Explainability
      │
Grounded Summary
      │
Interactive Dashboard
```

---

## Project Structure

```text
ETD-Semantic-Retrieval-Framework/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── analytics.py
│   ├── evaluation.py
│   ├── search_engine.py
│   ├── explainability.py
│   └── ...
│
├── data/
│   └── etd_records.csv
│
├── results/
│   ├── model_comparison.csv
│   ├── ablation_study.csv
│   ├── sensitivity_analysis.csv
│   └── ...
│
├── figures/
│
└── paper/
```

---

## Methodology

The retrieval pipeline consists of:

1. Metadata preprocessing
2. Sentence embedding generation
3. Dense vector indexing using FAISS
4. Semantic similarity retrieval
5. Metadata filtering
6. Explainability generation
7. Grounded summary generation
8. Interactive analytics

---

## Technologies Used

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| Web Interface | Streamlit |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas |
| Visualisation | Plotly |
| Keyword Baseline | BM25 |

---

## Experimental Evaluation

The framework was evaluated using:

- Precision@K
- Recall@K
- MAP
- NDCG
- Query latency
- Embedding model comparison
- Ablation study
- Sensitivity analysis

Models evaluated:

- all-MiniLM-L6-v2
- all-mpnet-base-v2

---

## Running Locally

Clone the repository

```bash
git clone https://github.com/MrigankJaiswal-hub/ETD-Semantic-Retrieval-Framework.git
```

Move into the project

```bash
cd ETD-Semantic-Retrieval-Framework
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch the application

```bash
streamlit run app.py
```

---

## Live Demo

Streamlit Deployment

> https://YOUR-STREAMLIT-LINK.streamlit.app

---

## Research Contributions

This work introduces a lightweight semantic retrieval framework for ETD repositories by integrating:

- Dense semantic retrieval
- Explainable search
- Grounded summaries
- Metadata-aware filtering
- Interactive retrieval analytics
- Comprehensive retrieval evaluation

The implementation demonstrates how modern information retrieval techniques can improve scholarly discovery while maintaining efficient query performance.

---

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{jaiswal2026etd,
  author = {Mrigank Jaiswal},
  title = {Conversational Semantic Search for Electronic Theses and Dissertations Using Dense Retrieval and Explainable AI},
  booktitle = {ETD 2026 Symposium},
  year = {2026},
  note = {Submitted}
}
```

---

## Future Work

- Multilingual semantic retrieval
- Cross-lingual embeddings
- Hybrid dense–sparse retrieval
- Learning-to-rank
- Retrieval-Augmented Generation (RAG)
- Integration with DSpace, EPrints and Greenstone
- Large-scale institutional ETD repositories

---

## Author

**Mrigank Jaiswal**

B.Tech, Electronics and Communication Engineering

Central University of Jammu, India

Research Interests:

- Information Retrieval
- Semantic Search
- Digital Libraries
- Natural Language Processing
- Artificial Intelligence
- Retrieval-Augmented Generation

Email:

23beece10.ece@cujammu.ac.in

---

## License

This project is released under the MIT License.
