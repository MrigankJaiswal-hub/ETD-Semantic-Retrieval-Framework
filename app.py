import streamlit as st
import pandas as pd

from src.search_engine import ETDSearchEngine
from src.analytics import (
    render_retrieval_analytics_dashboard,
    render_semantic_embedding_visualization,
)
from src.evaluation import render_ir_evaluation_dashboard


st.set_page_config(
    page_title="ETD Semantic Discovery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.hero-card {
    padding: 2.3rem;
    border-radius: 26px;
    border: 1px solid #bfdbfe;
    background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
    margin-bottom: 2rem;
}
.main-title {
    font-size: 3rem;
    font-weight: 900;
    color: #0f172a;
    line-height: 1.12;
}
.subtitle {
    font-size: 1.15rem;
    color: #475569;
    margin-top: 1rem;
}
.pipeline-box {
    padding: 1rem 1.3rem;
    border-radius: 18px;
    border: 1px solid #bfdbfe;
    background: #f8fafc;
    font-size: 1rem;
    margin-top: 1.2rem;
}
.badge {
    display: inline-block;
    padding: 0.42rem 0.78rem;
    border-radius: 999px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    font-weight: 700;
    margin: 0.25rem;
}
.success-step {
    padding: 1rem;
    border-radius: 16px;
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
    color: #047857;
    text-align: center;
    font-weight: 800;
}
.top-match {
    padding: 1rem 1.4rem;
    border-radius: 18px;
    border: 1px solid #bfdbfe;
    background: #f8fafc;
    margin-top: 1rem;
}
.abstract-box {
    padding: 1.1rem 1.4rem;
    border-radius: 14px;
    background: #eaf3ff;
    color: #0759b8;
    font-size: 1.02rem;
    line-height: 1.6;
}
.footer-card {
    padding: 1.8rem;
    border-radius: 24px;
    border: 1px solid #bfdbfe;
    background: #f8fafc;
    margin-top: 2rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def render_badges(items):
    if not items:
        st.write("No items available.")
        return

    html = " ".join(
        f'<span class="badge">{str(item).title()}</span>'
        for item in items
        if str(item).strip()
    )
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    return ETDSearchEngine()


engine = load_engine()
df = engine.df

for key, default in {
    "query_text": "",
    "recent_searches": [],
    "last_query": "",
    "last_results": [],
    "last_timings": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


st.sidebar.header("Search Filters")

departments = ["All"] + engine.get_departments()
department_filter = st.sidebar.selectbox("Department", departments)

year_min, year_max = engine.get_year_range()

year_range = st.sidebar.slider(
    "Publication Year",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

top_k = st.sidebar.slider("Top K Results", 3, 10, 5)

similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.00,
    max_value=1.00,
    value=0.00,
    step=0.05,
)

sort_by = st.sidebar.selectbox(
    "Sort Results By",
    ["Highest Similarity", "Newest First", "Oldest First"],
)

st.sidebar.divider()
st.sidebar.header("Dataset Coverage")
st.sidebar.write(f"**Total ETDs:** {len(df)}")
st.sidebar.write(f"**Research domains:** {df['department'].nunique()}")
st.sidebar.write(f"**Publication years:** {year_min}–{year_max}")
st.sidebar.write("**Embedding size:** 384 dimensions")

st.sidebar.divider()
st.sidebar.header("Prototype Configuration")
st.sidebar.write("**Embedding model:** all-MiniLM-L6-v2")
st.sidebar.write("**Embedding library:** Sentence Transformers")
st.sidebar.write("**Vector database:** FAISS")
st.sidebar.write("**Retrieval method:** Semantic similarity")
st.sidebar.write("**Similarity metric:** Cosine similarity")
st.sidebar.write("**Response type:** Grounded retrieval + IR evaluation")

if st.session_state.recent_searches:
    st.sidebar.divider()
    st.sidebar.header("Recent Searches")
    for i, q in enumerate(st.session_state.recent_searches[-6:][::-1]):
        if st.sidebar.button(q, key=f"recent_{i}", use_container_width=True):
            st.session_state.query_text = q
            st.rerun()


st.markdown(
    """
<div class="hero-card">
    <div class="main-title">📚 AI-powered Semantic Discovery<br>for ETD Repositories</div>
    <div class="subtitle">
        Search Electronic Theses and Dissertations using natural language, semantic ranking,
        grounded retrieval summaries, explainability, metadata filters, retrieval analytics,
        and information retrieval evaluation.
    </div>
    <div class="pipeline-box">
        <b>Prototype Pipeline:</b>
        🧠 Sentence Transformer → 📦 FAISS Vector Index → 🔎 Semantic Retrieval →
        🧩 Explainability → 📊 Analytics → 📐 IR Evaluation → 💬 Grounded Summary
    </div>
</div>
""",
    unsafe_allow_html=True,
)


st.header("Try example queries")

example_queries = [
    "AI in healthcare",
    "wireless communication",
    "digital libraries",
    "renewable energy",
    "OCR for scanned theses",
    "metadata quality",
    "RAG systems",
    "semantic search",
]

cols = st.columns(4)

for i, q in enumerate(example_queries):
    with cols[i % 4]:
        if st.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state.query_text = q
            st.rerun()


query = st.text_input(
    "🔍 Search ETDs using natural language:",
    placeholder="Example: Find recent theses on semantic search",
    key="query_text",
)

search_clicked = st.button("Search ETDs", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        clean_query = query.strip()

        if clean_query not in st.session_state.recent_searches:
            st.session_state.recent_searches.append(clean_query)

        progress = st.progress(0, text="Creating query embedding...")
        progress.progress(25, text="Searching FAISS vector index...")
        progress.progress(50, text="Ranking ETD records...")
        progress.progress(75, text="Generating grounded summary...")

        results, timings = engine.semantic_search(
            query=clean_query,
            top_k=top_k,
            department=department_filter,
            year_range=year_range,
            similarity_threshold=similarity_threshold,
            sort_by=sort_by,
        )

        progress.progress(100, text="Search complete.")

        st.session_state.last_query = clean_query
        st.session_state.last_results = results
        st.session_state.last_timings = timings


results = st.session_state.last_results
timings = st.session_state.last_timings
last_query = st.session_state.last_query

if results:
    scores = [float(item["score"]) for item in results]
    avg_similarity = sum(scores) / len(scores)
    highest_similarity = max(scores)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📄 Retrieved Results", f"{len(results)} / {len(df)}")
    m2.metric("⚡ Total Latency", f"{timings.get('total_time_ms', 0):.1f} ms")
    m3.metric("🧠 Avg Similarity", f"{avg_similarity:.3f}")
    m4.metric("🎯 Highest Similarity", f"{highest_similarity:.3f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Embedding Time", f"{timings.get('embedding_time_ms', 0):.1f} ms")
    m6.metric("FAISS Search Time", f"{timings.get('faiss_search_time_ms', 0):.1f} ms")
    m7.metric("Ranking Time", f"{timings.get('ranking_time_ms', 0):.1f} ms")
    m8.metric("Embedding Dimension", "384")

    st.header("Interactive Retrieval Pipeline")

    p1, p2, p3, p4 = st.columns(4)
    p1.markdown('<div class="success-step">① Query Embedded</div>', unsafe_allow_html=True)
    p2.markdown('<div class="success-step">② FAISS Search</div>', unsafe_allow_html=True)
    p3.markdown(f'<div class="success-step">③ Top {len(results)} Ranked</div>', unsafe_allow_html=True)
    p4.markdown('<div class="success-step">④ Summary Generated</div>', unsafe_allow_html=True)

    top = results[0]

    st.markdown(
        f"""
<div class="top-match">
🏆 <b>Top Semantic Match:</b> {top['title']}<br>
<b>Department:</b> {top['department']} |
<b>Year:</b> {top['year']} |
<b>Cosine Similarity:</b> {top['score']:.3f}
</div>
""",
        unsafe_allow_html=True,
    )

    st.header("Grounded Retrieval Summary")
    st.markdown(engine.grounded_summary(last_query, results))

    tabs = st.tabs(
        [
            "Ranked Results",
            "Explainability",
            "Keyword Baseline",
            "Analytics Dashboard",
            "Semantic Space",
            "IR Evaluation",
            "Compare Results",
        ]
    )

    with tabs[0]:
        st.header(f"Top {len(results)} Ranked Results")

        for item in results:
            with st.expander(
                f"🏅 Rank {item['rank']} | {item['title']} | "
                f"Cosine Similarity: {item['score']:.3f} | "
                f"{item['relevance_icon']} {item['relevance']}",
                expanded=(item["rank"] == 1),
            ):
                left, right = st.columns(2)

                with left:
                    st.subheader("Metadata")
                    st.write(f"📄 **Title:** {item['title']}")
                    st.write(f"📘 **Document Type:** {item['document_type']}")
                    st.write(f"🏛️ **Institution:** {item['institution']}")
                    st.write(f"📅 **Year:** {item['year']}")
                    st.write(f"🏫 **Department:** {item['department']}")

                with right:
                    st.subheader("Similarity")
                    st.write(f"🧠 **Cosine Similarity:** {item['score']:.3f}")
                    st.write(
                        f"📊 **Similarity Bar:** {item['similarity_bar']} "
                        f"{item['similarity_percent']}%"
                    )
                    st.write(
                        f"🎯 **Relevance Level:** "
                        f"{item['relevance_icon']} {item['relevance']}"
                    )

                    st.write("**Keywords:**")
                    render_badges(engine.split_keywords(item["keywords"]))

                st.subheader("Matched Concepts")
                render_badges(item.get("matched_concepts", []))

                st.subheader("Abstract")
                st.markdown(
                    f'<div class="abstract-box">{item["abstract"]}</div>',
                    unsafe_allow_html=True,
                )

    with tabs[1]:
        st.header("Semantic Similarity Explanation Panel")

        selected_title = st.selectbox(
            "Select a retrieved ETD to explain",
            [item["title"] for item in results],
        )

        selected = next(item for item in results if item["title"] == selected_title)
        explanation = engine.explain_result(last_query, selected)

        st.subheader("Why this result?")
        st.write(f"**Confidence:** {explanation['confidence']}")
        st.write(f"**Cosine Similarity:** {explanation['cosine_similarity']:.3f}")

        st.write("**Keyword overlap with query:**")
        if explanation["keyword_overlap"]:
            render_badges(explanation["keyword_overlap"])
        else:
            st.write("No direct keyword overlap. Retrieved mainly through semantic similarity.")

        st.write("**Keyword signals from ETD metadata:**")
        render_badges(explanation["keyword_signals"])

        st.info(explanation["explanation"])

    with tabs[2]:
        st.header("Keyword Search Baseline")

        keyword_results = engine.keyword_search(
            query=last_query,
            top_k=top_k,
            department=department_filter,
            year_range=year_range,
        )

        comparison_rows = []

        for i in range(max(len(keyword_results), len(results))):
            kr = keyword_results[i] if i < len(keyword_results) else {}
            sr = results[i] if i < len(results) else {}

            comparison_rows.append(
                {
                    "Rank": str(i + 1),
                    "Keyword Search Result": str(kr.get("title", "")),
                    "Keyword Score": f"{float(kr.get('keyword_score', 0)):.4f}" if kr else "",
                    "Semantic Search Result": str(sr.get("title", "")),
                    "Semantic Score": f"{float(sr.get('score', 0)):.4f}" if sr else "",
                }
            )

        st.dataframe(pd.DataFrame(comparison_rows).astype(str), use_container_width=True)

    with tabs[3]:
        render_retrieval_analytics_dashboard(results)

    with tabs[4]:
        render_semantic_embedding_visualization(
            query=last_query,
            results=results,
            engine=engine,
        )

    with tabs[5]:
        keyword_results = engine.keyword_search(
            query=last_query,
            top_k=top_k,
            department=department_filter,
            year_range=year_range,
        )

        render_ir_evaluation_dashboard(
            query=last_query,
            semantic_results=results,
            keyword_results=keyword_results,
            k=top_k,
        )

    with tabs[6]:
        st.header("Compare Two Retrieved ETDs")

        if len(results) >= 2:
            titles = [item["title"] for item in results]

            c1, c2 = st.columns(2)

            with c1:
                title_a = st.selectbox("Record A", titles, index=0)

            with c2:
                title_b = st.selectbox("Record B", titles, index=1)

            record_a = next(item for item in results if item["title"] == title_a)
            record_b = next(item for item in results if item["title"] == title_b)

            compare_df = pd.DataFrame(
                [
                    ["Title", str(record_a["title"]), str(record_b["title"])],
                    ["Year", str(record_a["year"]), str(record_b["year"])],
                    ["Department", str(record_a["department"]), str(record_b["department"])],
                    [
                        "Cosine Similarity",
                        f"{float(record_a['score']):.4f}",
                        f"{float(record_b['score']):.4f}",
                    ],
                    ["Keywords", str(record_a["keywords"]), str(record_b["keywords"])],
                    ["Abstract", str(record_a["abstract"]), str(record_b["abstract"])],
                ],
                columns=["Field", "Record A", "Record B"],
            ).astype(str)

            st.dataframe(compare_df, use_container_width=True)
        else:
            st.info("At least two results are required for comparison.")

else:
    st.info("Enter a query and click **Search ETDs** to begin.")


st.markdown(
    """
<div class="footer-card">
<b>Conversational Semantic Search for ETD Repositories</b><br><br>
<b>Embedding:</b> Sentence Transformers<br>
<b>Embedding Model:</b> all-MiniLM-L6-v2<br>
<b>Vector Index:</b> FAISS<br>
<b>Similarity:</b> Cosine Similarity<br>
<b>Dataset:</b> Curated ETD-style records<br>
<b>Response Generation:</b> Grounded retrieval over metadata and abstracts<br>
<b>Research Layer:</b> BM25 baseline comparison, Precision@K, Recall@K, NDCG@K, and MAP@K<br>
<b>Prototype Version:</b> v4.0 Research Evaluation<br>
<b>Year:</b> 2026<br><br>
<b>© 2026 Mrigank Jaiswal</b>
</div>
""",
    unsafe_allow_html=True,
)