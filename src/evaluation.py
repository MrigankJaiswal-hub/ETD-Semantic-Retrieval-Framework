# src/evaluation.py

from __future__ import annotations

from typing import Any, Dict, List
import math
import pandas as pd
import streamlit as st


def _score_to_relevance(score: float) -> int:
    if score >= 0.50:
        return 3
    if score >= 0.30:
        return 2
    if score >= 0.15:
        return 1
    return 0


def precision_at_k(results: List[Dict[str, Any]], k: int = 5, threshold: float = 0.30) -> float:
    top = results[:k]
    if not top:
        return 0.0
    relevant = sum(1 for r in top if float(r.get("score", 0)) >= threshold)
    return relevant / len(top)


def recall_at_k(results: List[Dict[str, Any]], k: int = 5, threshold: float = 0.30) -> float:
    relevant_total = sum(1 for r in results if float(r.get("score", 0)) >= threshold)
    if relevant_total == 0:
        return 0.0
    relevant_top = sum(1 for r in results[:k] if float(r.get("score", 0)) >= threshold)
    return relevant_top / relevant_total


def dcg_at_k(scores: List[float], k: int = 5) -> float:
    rels = [_score_to_relevance(s) for s in scores[:k]]
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(results: List[Dict[str, Any]], k: int = 5) -> float:
    scores = [float(r.get("score", 0)) for r in results]
    if not scores:
        return 0.0

    dcg = dcg_at_k(scores, k)
    ideal_scores = sorted(scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)

    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(results: List[Dict[str, Any]], k: int = 5, threshold: float = 0.30) -> float:
    hits = 0
    precision_sum = 0.0

    for i, r in enumerate(results[:k], start=1):
        if float(r.get("score", 0)) >= threshold:
            hits += 1
            precision_sum += hits / i

    return precision_sum / hits if hits > 0 else 0.0


def render_ir_evaluation_dashboard(
    query: str,
    semantic_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = 5,
) -> None:
    st.header("Information Retrieval Evaluation")

    if not semantic_results:
        st.info("Run a search first to view evaluation metrics.")
        return

    p_at_k = precision_at_k(semantic_results, k=k)
    r_at_k = recall_at_k(semantic_results, k=k)
    n_at_k = ndcg_at_k(semantic_results, k=k)
    m_at_k = map_at_k(semantic_results, k=k)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Precision@{k}", f"{p_at_k:.3f}")
    c2.metric(f"Recall@{k}", f"{r_at_k:.3f}")
    c3.metric(f"NDCG@{k}", f"{n_at_k:.3f}")
    c4.metric(f"MAP@{k}", f"{m_at_k:.3f}")

    st.markdown(
        """
These metrics estimate retrieval quality using similarity-based relevance labels.
A result with cosine similarity ≥ 0.30 is treated as relevant for prototype evaluation.
"""
    )

    rows = []

    max_len = max(len(semantic_results), len(keyword_results))

    for i in range(max_len):
        sr = semantic_results[i] if i < len(semantic_results) else {}
        kr = keyword_results[i] if i < len(keyword_results) else {}

        rows.append(
            {
                "Rank": i + 1,
                "Semantic Search Result": sr.get("title", ""),
                "Semantic Score": round(float(sr.get("score", 0)), 4) if sr else "",
                "BM25 Keyword Result": kr.get("title", ""),
                "BM25 Score": round(float(kr.get("keyword_score", 0)), 4) if kr else "",
            }
        )

    st.subheader("Semantic Search vs BM25 Baseline")
    st.dataframe(pd.DataFrame(rows).astype(str), use_container_width=True)

    st.subheader("Research Contribution Summary")

    st.success(
        f"""
For the query **“{query}”**, the system evaluates semantic retrieval against a BM25 keyword baseline.
The prototype demonstrates how dense sentence embeddings and FAISS-based vector search can retrieve
contextually relevant ETD records beyond exact keyword matching.

Research contributions:
1. Semantic ranking for ETD repository discovery.
2. Explainable retrieval through keyword overlap and metadata signals.
3. Retrieval analytics using similarity distribution, department spread, and timeline analysis.
4. Quantitative IR evaluation using Precision@K, Recall@K, NDCG@K, and MAP@K.
5. Comparative baseline analysis against BM25 keyword retrieval.
"""
    )