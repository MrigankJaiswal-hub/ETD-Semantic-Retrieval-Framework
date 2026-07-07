import os
import re
import time
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class SearchTiming:
    embedding_time_ms: float
    faiss_search_time_ms: float
    ranking_time_ms: float
    total_time_ms: float


class ETDSearchEngine:
    def __init__(
        self,
        data_path: str = "data/etd_records.csv",
        model_dir: str = "models",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model_name = model_name

        self.index_path = os.path.join(model_dir, "faiss.index")
        self.embeddings_path = os.path.join(model_dir, "embeddings.npy")
        self.metadata_path = os.path.join(model_dir, "metadata.pkl")

        os.makedirs(model_dir, exist_ok=True)

        self.df = self.load_dataset()
        self.model = SentenceTransformer(model_name)

        self.record_texts = self.build_record_texts()
        self.bm25 = self.build_bm25()

        self.index, self.embeddings = self.load_or_build_index()

    # -------------------------
    # Data loading
    # -------------------------

    def load_dataset(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        df = pd.read_csv(self.data_path)

        required = ["id", "title", "abstract", "keywords", "year", "department"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.fillna("")
        df["id"] = df["id"].astype(str)
        df["title"] = df["title"].astype(str)
        df["abstract"] = df["abstract"].astype(str)
        df["keywords"] = df["keywords"].astype(str)
        df["department"] = df["department"].astype(str)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

        return df

    def build_record_texts(self) -> List[str]:
        texts = []

        for _, row in self.df.iterrows():
            text = (
                f"Title: {row['title']}. "
                f"Abstract: {row['abstract']}. "
                f"Keywords: {row['keywords']}. "
                f"Department: {row['department']}. "
                f"Year: {row['year']}."
            )
            texts.append(text)

        return texts

    # -------------------------
    # Embedding helper
    # -------------------------

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Single embedding API for the whole project.

        Used by:
        - semantic search
        - analytics.py
        - visualizations.py
        - t-SNE/PCA embedding plots
        - similarity heatmaps
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    # -------------------------
    # Indexing
    # -------------------------

    def load_or_build_index(self):
        if (
            os.path.exists(self.index_path)
            and os.path.exists(self.embeddings_path)
            and os.path.exists(self.metadata_path)
        ):
            index = faiss.read_index(self.index_path)
            embeddings = np.load(self.embeddings_path)

            with open(self.metadata_path, "rb") as f:
                saved_columns = pickle.load(f)

            if saved_columns == list(self.df.columns):
                return index, embeddings

        return self.build_index()

    def build_index(self):
        embeddings = self.encode_texts(self.record_texts)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, self.index_path)
        np.save(self.embeddings_path, embeddings)

        with open(self.metadata_path, "wb") as f:
            pickle.dump(list(self.df.columns), f)

        return index, embeddings

    # -------------------------
    # BM25 keyword baseline
    # -------------------------

    def build_bm25(self):
        tokenized = [self.tokenize(text) for text in self.record_texts]
        return BM25Okapi(tokenized)

    # -------------------------
    # Utility functions
    # -------------------------

    @staticmethod
    def normalize(text: Any) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def tokenize(cls, text: Any) -> List[str]:
        return cls.normalize(text).split()

    @staticmethod
    def split_keywords(keywords: Any) -> List[str]:
        parts = re.split(r"[;,|]", str(keywords))
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def relevance_label(score: float) -> str:
        if score >= 0.50:
            return "High"
        if score >= 0.30:
            return "Medium"
        return "Low"

    @staticmethod
    def relevance_icon(score: float) -> str:
        if score >= 0.50:
            return "🟢"
        if score >= 0.30:
            return "🟡"
        return "🔴"

    @staticmethod
    def similarity_bar(score: float, blocks: int = 10) -> str:
        score = max(0.0, min(float(score), 1.0))
        filled = int(round(score * blocks))
        empty = blocks - filled
        return "█" * filled + "░" * empty

    def get_departments(self) -> List[str]:
        return sorted(self.df["department"].dropna().unique().tolist())

    def get_year_range(self) -> Tuple[int, int]:
        return int(self.df["year"].min()), int(self.df["year"].max())

    # -------------------------
    # Result formatting
    # -------------------------

    def row_to_result(self, row_idx: int, score: float, rank: int, query: str) -> Dict:
        row = self.df.iloc[row_idx]

        result = {
            "id": str(row["id"]),
            "title": str(row["title"]),
            "abstract": str(row["abstract"]),
            "keywords": str(row["keywords"]),
            "year": int(row["year"]),
            "department": str(row["department"]),
            "score": float(score),
            "similarity": float(score),
            "rank": int(rank),
            "relevance": self.relevance_label(float(score)),
            "relevance_icon": self.relevance_icon(float(score)),
            "similarity_bar": self.similarity_bar(float(score)),
            "similarity_percent": round(float(score) * 100, 1),
            "document_type": "Master Thesis",
            "institution": "Prototype Dataset",
        }

        result["matched_concepts"] = self.matched_concepts(query, result)
        return result

    def matched_concepts(self, query: str, result: Dict) -> List[str]:
        query_terms = set(self.tokenize(query))

        stopwords = {
            "find", "show", "thesis", "theses", "etd", "etds", "research",
            "paper", "papers", "on", "in", "for", "using", "based", "related",
            "about", "the", "and", "or", "of", "to", "a", "an",
        }

        query_terms = {t for t in query_terms if t not in stopwords and len(t) > 2}

        combined_text = self.normalize(
            f"{result['title']} {result['abstract']} "
            f"{result['keywords']} {result['department']}"
        )

        concepts = []

        for term in query_terms:
            if term in combined_text:
                concepts.append(term.title())

        for kw in self.split_keywords(result["keywords"]):
            concepts.append(kw.title())

        unique = []
        seen = set()

        for c in concepts:
            key = c.lower()
            if key not in seen:
                unique.append(c)
                seen.add(key)

        return unique[:8]

    # -------------------------
    # Filters
    # -------------------------

    def passes_filters(
        self,
        row_idx: int,
        department: str = "All",
        year_range: Optional[Tuple[int, int]] = None,
        similarity_threshold: float = 0.0,
        score: float = 0.0,
    ) -> bool:
        row = self.df.iloc[row_idx]

        if float(score) < float(similarity_threshold):
            return False

        if department and department != "All":
            if str(row["department"]).lower() != department.lower():
                return False

        if year_range:
            y1, y2 = year_range
            if not (int(y1) <= int(row["year"]) <= int(y2)):
                return False

        return True

    # -------------------------
    # Semantic search
    # -------------------------

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        department: str = "All",
        year_range: Optional[Tuple[int, int]] = None,
        similarity_threshold: float = 0.0,
        sort_by: str = "Highest Similarity",
    ) -> Tuple[List[Dict], Dict]:
        start_total = time.perf_counter()

        start_embed = time.perf_counter()
        query_embedding = self.encode_texts([query])
        embedding_time = (time.perf_counter() - start_embed) * 1000

        search_k = min(len(self.df), max(top_k * 10, 50))

        start_faiss = time.perf_counter()
        scores, indices = self.index.search(query_embedding, search_k)
        faiss_time = (time.perf_counter() - start_faiss) * 1000

        start_rank = time.perf_counter()

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            if not self.passes_filters(
                row_idx=idx,
                department=department,
                year_range=year_range,
                similarity_threshold=similarity_threshold,
                score=float(score),
            ):
                continue

            result = self.row_to_result(
                row_idx=idx,
                score=float(score),
                rank=len(results) + 1,
                query=query,
            )
            results.append(result)

            if len(results) >= top_k:
                break

        if sort_by == "Newest First":
            results = sorted(results, key=lambda r: r["year"], reverse=True)
        elif sort_by == "Oldest First":
            results = sorted(results, key=lambda r: r["year"])
        else:
            results = sorted(results, key=lambda r: r["score"], reverse=True)

        for i, item in enumerate(results, start=1):
            item["rank"] = i

        ranking_time = (time.perf_counter() - start_rank) * 1000
        total_time = (time.perf_counter() - start_total) * 1000

        timings = {
            "embedding_time_ms": embedding_time,
            "faiss_search_time_ms": faiss_time,
            "ranking_time_ms": ranking_time,
            "total_time_ms": total_time,
        }

        return results, timings

    # -------------------------
    # Keyword search baseline
    # -------------------------

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        department: str = "All",
        year_range: Optional[Tuple[int, int]] = None,
    ) -> List[Dict]:
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1]

        results = []

        max_score = float(max(scores)) if len(scores) else 1.0
        if max_score == 0:
            max_score = 1.0

        for idx in ranked_indices:
            raw_score = float(scores[idx])
            norm_score = raw_score / max_score

            if not self.passes_filters(
                row_idx=idx,
                department=department,
                year_range=year_range,
                similarity_threshold=0.0,
                score=norm_score,
            ):
                continue

            result = self.row_to_result(
                row_idx=idx,
                score=norm_score,
                rank=len(results) + 1,
                query=query,
            )
            result["keyword_score"] = raw_score
            result["normalized_keyword_score"] = norm_score
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    # -------------------------
    # Hybrid search
    # -------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.75,
        keyword_weight: float = 0.25,
        department: str = "All",
        year_range: Optional[Tuple[int, int]] = None,
        similarity_threshold: float = 0.0,
    ) -> List[Dict]:
        semantic_results, _ = self.semantic_search(
            query=query,
            top_k=max(top_k * 3, 20),
            department=department,
            year_range=year_range,
            similarity_threshold=similarity_threshold,
        )

        keyword_results = self.keyword_search(
            query=query,
            top_k=max(top_k * 3, 20),
            department=department,
            year_range=year_range,
        )

        combined = {}

        for item in semantic_results:
            combined[item["id"]] = item.copy()
            combined[item["id"]]["semantic_score"] = item["score"]
            combined[item["id"]]["keyword_score_norm"] = 0.0

        for item in keyword_results:
            item_id = item["id"]

            if item_id not in combined:
                combined[item_id] = item.copy()
                combined[item_id]["semantic_score"] = item.get("score", 0.0)

            combined[item_id]["keyword_score_norm"] = item.get(
                "normalized_keyword_score", 0.0
            )

        final = []

        for item in combined.values():
            semantic_score = float(item.get("semantic_score", 0.0))
            keyword_score = float(item.get("keyword_score_norm", 0.0))

            hybrid_score = semantic_weight * semantic_score + keyword_weight * keyword_score

            item["hybrid_score"] = hybrid_score
            item["score"] = semantic_score
            item["similarity"] = semantic_score
            item["matched_concepts"] = self.matched_concepts(query, item)
            final.append(item)

        final = sorted(final, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]

        for i, item in enumerate(final, start=1):
            item["rank"] = i

        return final

    # -------------------------
    # Explanation
    # -------------------------

    def explain_result(self, query: str, item: Dict) -> Dict:
        query_terms = set(self.tokenize(query))
        text = self.normalize(
            f"{item.get('title', '')} {item.get('abstract', '')} "
            f"{item.get('keywords', '')} {item.get('department', '')}"
        )

        overlap = [t.title() for t in query_terms if t in text and len(t) > 2]
        keywords = [kw.title() for kw in self.split_keywords(item.get("keywords", ""))]

        score = float(item.get("score", 0.0))

        if score >= 0.50:
            confidence = "Strong semantic match"
        elif score >= 0.30:
            confidence = "Moderate semantic match"
        else:
            confidence = "Weak semantic match"

        return {
            "confidence": confidence,
            "cosine_similarity": score,
            "keyword_overlap": overlap,
            "keyword_signals": keywords,
            "matched_concepts": item.get("matched_concepts", []),
            "explanation": (
                "This record was ranked using cosine similarity between the query embedding "
                "and the ETD metadata/abstract embedding. Keyword overlap is shown only as "
                "an interpretability aid; the final ranking is based on semantic similarity."
            ),
        }

    # -------------------------
    # Summary
    # -------------------------

    def grounded_summary(self, query: str, results: List[Dict]) -> str:
        if not results:
            return (
                f"For the query **“{query}”**, no closely related ETD records were retrieved. "
                "Try using broader terms, reducing filters, or lowering the similarity threshold."
            )

        top = results[0]
        related = results[1:4]

        summary = (
            f"For the query **“{query}”**, the prototype retrieved ETD records that appear "
            f"contextually related to the topic.\n\n"
            f"The highest-ranked result is **{top['title']}** from the "
            f"**{top['department']}** department, published in **{top['year']}**. "
            f"It was selected because its title, abstract, keywords, and metadata are "
            f"semantically close to the query.\n\n"
            f"> {top['abstract']}\n\n"
        )

        if related:
            summary += "Other related records include:\n\n"
            for item in related:
                summary += f"- **{item['title']}** ({item['year']}, {item['department']})\n"

        summary += (
            "\nThis response is generated only from retrieved ETD metadata and abstracts. "
            "It should be used as an initial discovery aid rather than a substitute for reading the full thesis."
        )

        return summary


if __name__ == "__main__":
    engine = ETDSearchEngine()
    results, timings = engine.semantic_search("AI in healthcare", top_k=5)

    print("Timings:", timings)

    for item in results:
        print(item["rank"], item["title"], item["score"])

    print(engine.grounded_summary("AI in healthcare", results))