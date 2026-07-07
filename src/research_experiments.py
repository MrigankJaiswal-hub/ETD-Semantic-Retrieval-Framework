import os
import time
import math
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import faiss


DATA_PATH = "data/etd_records.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


TEST_QUERIES = [
    {
        "query": "AI in healthcare",
        "relevant_terms": ["ai", "machine learning", "healthcare", "clinical", "medical", "diagnosis"],
    },
    {
        "query": "wireless communication",
        "relevant_terms": ["wireless", "5g", "communication", "mimo", "network", "signal"],
    },
    {
        "query": "digital libraries",
        "relevant_terms": ["digital library", "repository", "metadata", "academic", "document"],
    },
    {
        "query": "renewable energy storage",
        "relevant_terms": ["renewable", "energy", "storage", "battery", "solar"],
    },
    {
        "query": "OCR for scanned theses",
        "relevant_terms": ["ocr", "scanned", "digitisation", "text recognition", "document"],
    },
    {
        "query": "semantic search",
        "relevant_terms": ["semantic", "search", "retrieval", "embedding", "similarity"],
    },
]


def load_data():
    df = pd.read_csv(DATA_PATH).fillna("")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    df["record_text"] = (
        "Title: " + df["title"].astype(str) + ". "
        + "Abstract: " + df["abstract"].astype(str) + ". "
        + "Keywords: " + df["keywords"].astype(str) + ". "
        + "Department: " + df["department"].astype(str) + ". "
        + "Year: " + df["year"].astype(str)
    )

    return df


def is_relevant(record, relevant_terms):
    text = (
        str(record.get("title", "")) + " "
        + str(record.get("abstract", "")) + " "
        + str(record.get("keywords", "")) + " "
        + str(record.get("department", ""))
    ).lower()

    return any(term.lower() in text for term in relevant_terms)


def precision_at_k(results, relevant_terms, k):
    top = results[:k]
    if not top:
        return 0.0

    relevant = sum(1 for r in top if is_relevant(r, relevant_terms))
    return relevant / len(top)


def recall_at_k(results, all_records, relevant_terms, k):
    total_relevant = sum(1 for _, r in all_records.iterrows() if is_relevant(r, relevant_terms))
    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(1 for r in results[:k] if is_relevant(r, relevant_terms))
    return retrieved_relevant / total_relevant


def average_precision_at_k(results, relevant_terms, k):
    hits = 0
    score_sum = 0.0

    for i, r in enumerate(results[:k], start=1):
        if is_relevant(r, relevant_terms):
            hits += 1
            score_sum += hits / i

    return score_sum / hits if hits > 0 else 0.0


def dcg_at_k(results, relevant_terms, k):
    dcg = 0.0

    for i, r in enumerate(results[:k], start=1):
        rel = 1 if is_relevant(r, relevant_terms) else 0
        dcg += rel / math.log2(i + 1)

    return dcg


def ndcg_at_k(results, all_records, relevant_terms, k):
    dcg = dcg_at_k(results, relevant_terms, k)

    total_relevant = sum(1 for _, r in all_records.iterrows() if is_relevant(r, relevant_terms))
    ideal_relevant = min(total_relevant, k)

    if ideal_relevant == 0:
        return 0.0

    ideal_results = [{"dummy": 1} for _ in range(ideal_relevant)]
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_relevant + 1))

    return dcg / idcg if idcg > 0 else 0.0


def build_dense_index(df, model_name):
    model = SentenceTransformer(model_name)

    start_embed = time.perf_counter()
    embeddings = model.encode(
        df["record_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    embedding_time = (time.perf_counter() - start_embed) * 1000

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index, embedding_time, embeddings.shape[1]


def dense_search(df, model, index, query, top_k=10, department=None, year_range=None, threshold=0.0):
    start = time.perf_counter()

    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, ids = index.search(q_emb, min(len(df), max(top_k * 10, 50)))

    results = []

    for score, idx in zip(scores[0], ids[0]):
        row = df.iloc[int(idx)].to_dict()
        row["score"] = float(score)

        if row["score"] < threshold:
            continue

        if department and department != "All":
            if str(row["department"]).lower() != department.lower():
                continue

        if year_range:
            if not (year_range[0] <= int(row["year"]) <= year_range[1]):
                continue

        results.append(row)

        if len(results) >= top_k:
            break

    latency = (time.perf_counter() - start) * 1000

    return results, latency


def bm25_search(df, query, top_k=10):
    tokenised = [str(t).lower().split() for t in df["record_text"].tolist()]
    bm25 = BM25Okapi(tokenised)

    scores = bm25.get_scores(query.lower().split())
    ranked = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in ranked:
        row = df.iloc[int(idx)].to_dict()
        row["score"] = float(scores[idx])
        results.append(row)

    return results


def run_model_comparison():
    df = load_data()

    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
    ]

    rows = []

    for model_name in models:
        model, index, build_time, dim = build_dense_index(df, model_name)

        for item in TEST_QUERIES:
            query = item["query"]
            relevant_terms = item["relevant_terms"]

            results, latency = dense_search(df, model, index, query, top_k=10)

            rows.append(
                {
                    "model": model_name,
                    "embedding_dimension": dim,
                    "query": query,
                    "precision_at_5": precision_at_k(results, relevant_terms, 5),
                    "recall_at_5": recall_at_k(results, df, relevant_terms, 5),
                    "map_at_5": average_precision_at_k(results, relevant_terms, 5),
                    "ndcg_at_5": ndcg_at_k(results, df, relevant_terms, 5),
                    "precision_at_10": precision_at_k(results, relevant_terms, 10),
                    "recall_at_10": recall_at_k(results, df, relevant_terms, 10),
                    "map_at_10": average_precision_at_k(results, relevant_terms, 10),
                    "ndcg_at_10": ndcg_at_k(results, df, relevant_terms, 10),
                    "query_latency_ms": round(latency, 2),
                    "index_build_time_ms": round(build_time, 2),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    summary = (
        out.groupby("model")
        .agg(
            precision_at_5=("precision_at_5", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            map_at_5=("map_at_5", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            precision_at_10=("precision_at_10", "mean"),
            recall_at_10=("recall_at_10", "mean"),
            map_at_10=("map_at_10", "mean"),
            ndcg_at_10=("ndcg_at_10", "mean"),
            avg_query_latency_ms=("query_latency_ms", "mean"),
            avg_index_build_time_ms=("index_build_time_ms", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(os.path.join(RESULTS_DIR, "model_comparison_summary.csv"), index=False)

    print("\nModel comparison saved.")
    print(summary)


def run_ablation_study():
    df = load_data()
    model, index, _, _ = build_dense_index(df, "all-MiniLM-L6-v2")

    rows = []

    variants = [
        {
            "variant": "Semantic only",
            "department": None,
            "year_range": None,
            "threshold": 0.0,
            "summary_enabled": False,
            "explainability_enabled": False,
        },
        {
            "variant": "Semantic + metadata filtering",
            "department": "Computer Science",
            "year_range": None,
            "threshold": 0.0,
            "summary_enabled": False,
            "explainability_enabled": False,
        },
        {
            "variant": "Semantic + threshold filtering",
            "department": None,
            "year_range": None,
            "threshold": 0.30,
            "summary_enabled": False,
            "explainability_enabled": False,
        },
        {
            "variant": "Full framework",
            "department": None,
            "year_range": None,
            "threshold": 0.0,
            "summary_enabled": True,
            "explainability_enabled": True,
        },
    ]

    for variant in variants:
        for item in TEST_QUERIES:
            query = item["query"]
            relevant_terms = item["relevant_terms"]

            results, latency = dense_search(
                df,
                model,
                index,
                query,
                top_k=10,
                department=variant["department"],
                year_range=variant["year_range"],
                threshold=variant["threshold"],
            )

            rows.append(
                {
                    "variant": variant["variant"],
                    "query": query,
                    "precision_at_5": precision_at_k(results, relevant_terms, 5),
                    "recall_at_5": recall_at_k(results, df, relevant_terms, 5),
                    "map_at_5": average_precision_at_k(results, relevant_terms, 5),
                    "ndcg_at_5": ndcg_at_k(results, df, relevant_terms, 5),
                    "retrieved_count": len(results),
                    "query_latency_ms": round(latency, 2),
                    "summary_enabled": variant["summary_enabled"],
                    "explainability_enabled": variant["explainability_enabled"],
                    "metadata_filtering": variant["department"] is not None or variant["year_range"] is not None,
                    "threshold": variant["threshold"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "ablation_study.csv"), index=False)

    summary = (
        out.groupby("variant")
        .agg(
            precision_at_5=("precision_at_5", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            map_at_5=("map_at_5", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            avg_retrieved_count=("retrieved_count", "mean"),
            avg_query_latency_ms=("query_latency_ms", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(os.path.join(RESULTS_DIR, "ablation_study_summary.csv"), index=False)

    print("\nAblation study saved.")
    print(summary)


def run_sensitivity_analysis():
    df = load_data()
    model, index, _, _ = build_dense_index(df, "all-MiniLM-L6-v2")

    rows = []

    top_k_values = [3, 5, 10, 15, 20]
    thresholds = [0.00, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    for top_k in top_k_values:
        for threshold in thresholds:
            for item in TEST_QUERIES:
                query = item["query"]
                relevant_terms = item["relevant_terms"]

                results, latency = dense_search(
                    df,
                    model,
                    index,
                    query,
                    top_k=top_k,
                    threshold=threshold,
                )

                rows.append(
                    {
                        "top_k": top_k,
                        "similarity_threshold": threshold,
                        "query": query,
                        "precision": precision_at_k(results, relevant_terms, top_k),
                        "recall": recall_at_k(results, df, relevant_terms, top_k),
                        "map": average_precision_at_k(results, relevant_terms, top_k),
                        "ndcg": ndcg_at_k(results, df, relevant_terms, top_k),
                        "retrieved_count": len(results),
                        "query_latency_ms": round(latency, 2),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "sensitivity_analysis.csv"), index=False)

    summary = (
        out.groupby(["top_k", "similarity_threshold"])
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            map=("map", "mean"),
            ndcg=("ndcg", "mean"),
            avg_retrieved_count=("retrieved_count", "mean"),
            avg_query_latency_ms=("query_latency_ms", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(os.path.join(RESULTS_DIR, "sensitivity_analysis_summary.csv"), index=False)

    print("\nSensitivity analysis saved.")
    print(summary)


def run_all_experiments():
    run_model_comparison()
    run_ablation_study()
    run_sensitivity_analysis()


if __name__ == "__main__":
    run_all_experiments()