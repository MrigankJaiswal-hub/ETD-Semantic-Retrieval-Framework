# src/analytics.py

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}


def _safe_get(item: Dict[str, Any], key: str, default: Any = "") -> Any:
    value = item.get(key, default)
    return default if value is None else value


def _to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results, start=1):
        rows.append(
            {
                "Rank": int(r.get("rank", i)),
                "Title": str(_safe_get(r, "title")),
                "Department": str(_safe_get(r, "department")),
                "Year": int(r.get("year", 0)) if str(r.get("year", "")).isdigit() else str(r.get("year", "")),
                "Cosine Similarity": float(r.get("similarity", r.get("score", 0.0))),
                "Keywords": str(_safe_get(r, "keywords")),
                "Abstract": str(_safe_get(r, "abstract")),
            }
        )
    return pd.DataFrame(rows)


def _clean_plotly_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=70, b=35),
        font=dict(family="Arial", size=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _show_chart(fig: go.Figure) -> None:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


def _encode_texts(engine: Any, texts: List[str]) -> np.ndarray:
    if hasattr(engine, "encode_texts"):
        return np.asarray(engine.encode_texts(texts))

    if hasattr(engine, "model"):
        return np.asarray(
            engine.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

    raise AttributeError("Engine must provide either encode_texts() or model.encode().")


def render_retrieval_analytics_dashboard(results: List[Dict[str, Any]]) -> None:
    if not results:
        st.info("Run a search first to view retrieval analytics.")
        return

    df = _to_dataframe(results)

    st.header("Retrieval Analytics Dashboard")

    top_score = float(df["Cosine Similarity"].max())
    avg_score = float(df["Cosine Similarity"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieved Results", len(df))
    c2.metric("Top Similarity", f"{top_score:.3f}")
    c3.metric("Average Similarity", f"{avg_score:.3f}")
    c4.metric("Departments", df["Department"].nunique())

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=top_score * 100,
                number={"suffix": "%"},
                title={"text": "Top Match Similarity"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563eb", "thickness": 0.35},
                    "steps": [
                        {"range": [0, 30], "color": "#fee2e2"},
                        {"range": [30, 60], "color": "#fef3c7"},
                        {"range": [60, 100], "color": "#dcfce7"},
                    ],
                },
            )
        )
        _show_chart(_clean_plotly_layout(fig, height=360))

    with col2:
        dept_counts = df["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]

        fig = px.bar(
            dept_counts,
            x="Count",
            y="Department",
            orientation="h",
            title="Department Distribution in Retrieved Results",
            text="Count",
        )
        fig.update_yaxes(categoryorder="total ascending")
        _show_chart(_clean_plotly_layout(fig, height=360))

    col3, col4 = st.columns(2)

    with col3:
        year_counts = df["Year"].value_counts().sort_index().reset_index()
        year_counts.columns = ["Year", "Count"]

        fig = px.line(
            year_counts,
            x="Year",
            y="Count",
            markers=True,
            title="Publication Timeline of Retrieved Results",
        )
        _show_chart(_clean_plotly_layout(fig, height=360))

    with col4:
        keyword_counter = Counter()
        for kw_text in df["Keywords"].fillna(""):
            for kw in str(kw_text).replace(",", ";").split(";"):
                kw = kw.strip().lower()
                if kw:
                    keyword_counter[kw] += 1

        keyword_df = pd.DataFrame(
            keyword_counter.most_common(12),
            columns=["Keyword", "Frequency"],
        )

        if not keyword_df.empty:
            fig = px.bar(
                keyword_df,
                x="Frequency",
                y="Keyword",
                orientation="h",
                title="Top Keywords in Retrieved Results",
                text="Frequency",
            )
            fig.update_yaxes(categoryorder="total ascending")
            _show_chart(_clean_plotly_layout(fig, height=420))
        else:
            st.info("No keyword metadata available.")

    st.subheader("Similarity Score Distribution")

    df_bar = df.copy()
    df_bar["Similarity Label"] = df_bar["Cosine Similarity"].round(3).astype(str)

    fig = px.bar(
        df_bar,
        x="Rank",
        y="Cosine Similarity",
        text="Similarity Label",
        hover_data=["Title", "Department", "Year"],
        title="Similarity Distribution Across Retrieved Results",
    )
    _show_chart(_clean_plotly_layout(fig, height=380))

    st.subheader("Similarity Heatmap")

    heatmap_df = df[["Title", "Cosine Similarity"]].copy()
    heatmap_df["Short Title"] = heatmap_df["Title"].str.slice(0, 35)

    fig = px.imshow(
        [heatmap_df["Cosine Similarity"].tolist()],
        labels={"x": "Retrieved ETD", "y": "", "color": "Cosine Similarity"},
        x=heatmap_df["Short Title"].tolist(),
        y=["Query similarity"],
        text_auto=".3f",
        aspect="auto",
        title="Query-to-Result Similarity Heatmap",
    )
    _show_chart(_clean_plotly_layout(fig, height=320))

    st.subheader("Department Comparison Dashboard")

    dept_stats = (
        df.groupby("Department")
        .agg(
            Count=("Title", "count"),
            Average_Similarity=("Cosine Similarity", "mean"),
            Max_Similarity=("Cosine Similarity", "max"),
        )
        .reset_index()
    )

    dept_stats["Average_Similarity"] = dept_stats["Average_Similarity"].round(3)
    dept_stats["Max_Similarity"] = dept_stats["Max_Similarity"].round(3)

    st.dataframe(dept_stats.astype(str), width="stretch")


def render_semantic_embedding_visualization(
    query: str,
    results: List[Dict[str, Any]],
    engine: Any,
    method: str = "tsne",
) -> None:
    if not results:
        st.info("Run a search first to view semantic embedding space.")
        return

    st.header("t-SNE Semantic Embedding Visualization")

    texts = [query]
    labels = ["Query"]
    departments = ["Query"]
    ranks = ["Query"]

    for i, r in enumerate(results, start=1):
        text = " ".join(
            [
                str(r.get("title", "")),
                str(r.get("keywords", "")),
                str(r.get("abstract", "")),
                str(r.get("department", "")),
            ]
        )
        texts.append(text)
        labels.append(f"Rank {i}")
        departments.append(str(r.get("department", "Unknown")))
        ranks.append(f"Rank {i}")

    embeddings = _encode_texts(engine, texts)

    if len(texts) < 4 or method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        title = "PCA Semantic Embedding Space of Query and Retrieved ETDs"
    else:
        perplexity = max(2, min(5, len(texts) - 1))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="random",
            learning_rate="auto",
        )
        coords = reducer.fit_transform(embeddings)
        title = "t-SNE Semantic Embedding Space of Query and Retrieved ETDs"

    plot_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "Label": labels,
            "Department": departments,
            "Rank": ranks,
            "Text": [query] + [str(r.get("title", "")) for r in results],
            "Size": [18 if label == "Query" else 12 for label in labels],
        }
    )

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="Department",
        symbol="Rank",
        size="Size",
        hover_data=["Label", "Text", "Department"],
        title=title,
    )

    _show_chart(_clean_plotly_layout(fig, height=650))

    st.caption(
        "This visualization projects the query and retrieved ETD embeddings into 2D space. "
        "Nearby points indicate stronger semantic proximity in the embedding space."
    )