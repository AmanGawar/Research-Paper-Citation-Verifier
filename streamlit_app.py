import streamlit as st
import os
import tempfile
import uuid
import re
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modular RAG pipeline functions from backend
from main import (
    extract_text_from_pdf,
    extract_citations,
    extract_claims_from_sentences,
    chunk_text,
    build_faiss_index,
    build_chroma_collection,
    build_langchain_retriever,
    get_openai_verdict,
    compute_evaluation_stats,
    faiss_search,
    chroma_search,
    demo_result
)

# Set page configuration
st.set_page_config(
    page_title="CitationVerifier - RAG Citation Checker",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS for premium styling
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Font style definitions */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Styling Streamlit UI buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: transform 0.1s, box-shadow 0.1s !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    }
    div.stButton > button:active {
        transform: translateY(1px);
    }
    
    /* Secondary/demo button styling */
    div.stButton > button[key="demo_btn"] {
        background: white !important;
        color: #6366f1 !important;
        border: 1px solid #c7d2fe !important;
    }
    div.stButton > button[key="demo_btn"]:hover {
        background: #f5f3ff !important;
        box-shadow: none !important;
    }
    
    /* Hide Default Streamlit Style Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Summary details outline fix */
    summary {
        outline: none;
        user-select: none;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Header
st.markdown("""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 25px; border-bottom: 1px solid #e2e8f0; padding-bottom: 20px;">
    <div style="width: 48px; height: 48px; border-radius: 12px; background: linear-gradient(135deg, #6366f1, #8b5cf6); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);">
        <span style="font-size: 24px; color: white;">🛡️</span>
    </div>
    <div>
        <h1 style="margin: 0; font-size: 26px; font-weight: 700; color: #0f172a; line-height: 1.2;">CitationVerifier</h1>
        <p style="margin: 0; font-size: 13px; color: #94a3b8; font-weight: 500;">RAG-powered academic citation checker</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar stack info & credentials input
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
    <div style="font-size: 22px;">🛠️</div>
    <div style="font-size: 16px; font-weight: 700; color: #1e293b;">Technology Stack</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.info("""
- **LangChain** (RAG pipeline)
- **ChromaDB** (Persistent vector store)
- **FAISS** (High-speed vector index)
- **SentenceTransformers** (Local embeddings)
- **OpenAI GPT-4o-mini** (LLM verdicts)
- **Evaluation Pipeline** (Faithfulness & Risk)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Configuration")

api_key_input = st.sidebar.text_input(
    "OpenAI API Key (Optional)",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
    help="Used for GPT-4o-mini verdict generation. If left blank, the app falls back to local heuristic verification."
)

if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

# Main execution logic using session state
if "results" not in st.session_state:
    st.session_state.results = None
if "stats" not in st.session_state:
    st.session_state.stats = None

# Step-by-step verification pipeline runner
def run_verification(main_file, cited_file):
    status_container = st.container()
    with status_container:
        st.write("### ⚙️ Pipeline Execution")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # UI Steps indicators
        steps = [
            "Text Extraction",
            "Citation Parsing",
            "Text Chunking",
            "FAISS Indexing",
            "ChromaDB Storage",
            "LangChain Retriever Setup",
            "Claim Verdict Generation",
            "Evaluation Metrics"
        ]
        
        step_list_placeholder = st.empty()
        
        def update_ui_steps(active_index):
            html = "<div style='display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 10px;'>"
            for idx, s in enumerate(steps):
                active = idx == active_index
                completed = idx < active_index
                bg = "#ede9fe" if active or completed else "#f1f5f9"
                color = "#6366f1" if active or completed else "#94a3b8"
                prefix = "✓ " if completed else "● " if active else ""
                weight = "bold" if active else "normal"
                html += f"<span style='font-size: 11px; padding: 3px 10px; border-radius: 20px; background: {bg}; color: {color}; font-weight: {weight};'>{prefix}{s}</span>"
            html += "</div>"
            step_list_placeholder.markdown(html, unsafe_allow_html=True)

        update_ui_steps(0)

    # Save uploads to temp files
    status_text.markdown("📂 *Saving uploaded PDF files...*")
    progress_bar.progress(5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
        tmp1.write(main_file.read())
        main_path = tmp1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
        tmp2.write(cited_file.read())
        cited_path = tmp2.name

    try:
        # Step 1: Text Extraction
        status_text.markdown("📄 *Extracting full text from PDFs (pdfplumber / pymupdf)...*")
        progress_bar.progress(15)
        main_text = extract_text_from_pdf(main_path)
        cited_text = extract_text_from_pdf(cited_path)

        # Step 2: Citation Parsing
        update_ui_steps(1)
        status_text.markdown("✂️ *Parsing inline citations and claims from main paper...*")
        progress_bar.progress(25)
        citations = extract_citations(main_text)
        if not citations:
            citations = {"[ALL]": re.split(r'(?<=[.!?])\s+', main_text)[:30]}

        # Step 3: Text Chunking
        update_ui_steps(2)
        status_text.markdown(f"📦 *Chunking cited paper text ({len(cited_text.split())} words)...*")
        progress_bar.progress(35)
        chunks = chunk_text(cited_text, chunk_size=500, overlap=80)

        # Step 4: FAISS Indexing
        update_ui_steps(3)
        status_text.markdown("⚡ *Building FAISS vector index (Inner Product / Cosine)...*")
        progress_bar.progress(45)
        faiss_index, _ = build_faiss_index(chunks)

        # Step 5: ChromaDB Storage
        update_ui_steps(4)
        status_text.markdown("🗄️ *Building ChromaDB vector store collection...*")
        progress_bar.progress(55)
        col_name = f"cited_{uuid.uuid4().hex[:8]}"
        chroma_col = build_chroma_collection(col_name, chunks)

        # Step 6: LangChain Retriever Setup
        update_ui_steps(5)
        status_text.markdown("🔗 *Configuring LangChain FAISS retriever object...*")
        progress_bar.progress(60)
        lc_retriever = build_langchain_retriever(chunks)

        # Step 7: Claim Verdict Generation
        update_ui_steps(6)
        status_text.markdown("🤖 *Verifying claims against retriever evidence (calling LLM)...*")
        progress_bar.progress(65)

        results = []
        citation_items = list(citations.items())[:20]  # Cap at 20 citation groups

        for i, (cite_key, sentences) in enumerate(citation_items):
            status_text.markdown(f"🤖 *Verifying claim group {i+1}/{len(citation_items)} ({cite_key})...*")
            progress = 65 + int((i / len(citation_items)) * 30)
            progress_bar.progress(progress)

            claims = extract_claims_from_sentences(sentences, cite_key)
            if not claims:
                claims = sentences[:1]

            for claim in claims[:2]:  # Max 2 claims per citation key
                # FAISS retrieval
                faiss_hits = faiss_search(faiss_index, chunks, claim, top_k=5)
                faiss_chunks = [c for c, _ in faiss_hits]
                faiss_score = faiss_hits[0][1] if faiss_hits else 0.0

                # ChromaDB retrieval
                chroma_hits = chroma_search(chroma_col, claim, top_k=3)
                chroma_chunks = [c for c, _ in chroma_hits]

                # Merge and deduplicate evidence chunks
                all_evidence = list(dict.fromkeys(faiss_chunks + chroma_chunks))[:5]

                # LangChain retriever (third pass cross-validation)
                try:
                    lc_docs = lc_retriever.invoke(claim)
                    lc_chunks = [d.page_content for d in lc_docs]
                    all_evidence = list(dict.fromkeys(all_evidence + lc_chunks))[:5]
                except Exception:
                    pass

                # Get OpenAI or heuristic verdict
                verdict_data = get_openai_verdict(claim, all_evidence, cite_key)

                results.append({
                    "claim": claim[:500],
                    "citation_key": cite_key,
                    "verdict": verdict_data.get("verdict", "INSUFFICIENT"),
                    "confidence": verdict_data.get("confidence", 0.0),
                    "evidence": verdict_data.get("key_evidence", "")[:400],
                    "explanation": verdict_data.get("explanation", "")[:300],
                    "similarity_score": round(faiss_score, 4),
                })

        # Step 8: Evaluation Metrics
        update_ui_steps(7)
        status_text.markdown("📊 *Computing evaluation metrics...*")
        progress_bar.progress(97)
        stats = compute_evaluation_stats(results)

        progress_bar.progress(100)
        status_container.empty()
        
        st.session_state.results = results
        st.session_state.stats = stats
        st.rerun()

    except Exception as e:
        status_container.empty()
        st.error(f"Verification process encountered an error: {e}")
    finally:
        # Clean up temporary PDF files
        for p in [main_path, cited_path]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass

# Render initial view with uploaders and demo option
if st.session_state.results is None:
    st.markdown("""
    <h2 style="margin: 0 0 6px 0; font-size: 18px; font-weight: 700; color: #0f172a;">Upload Research Papers</h2>
    <p style="margin: 0 0 20px 0; color: #64748b; font-size: 14px;">
        Upload the <strong>main paper</strong> (whose claims you want to verify) and the <strong>cited paper</strong> it references.
    </p>
    """, unsafe_allow_html=True)

    col_main, col_cited = st.columns(2)
    with col_main:
        st.markdown("<div style='font-size: 12px; font-weight: 600; color: #6366f1; margin-bottom: 6px;'>📄 MAIN PAPER (with citations)</div>", unsafe_allow_html=True)
        main_paper_file = st.file_uploader("Upload Main Paper", type=["pdf"], label_visibility="collapsed")
    with col_cited:
        st.markdown("<div style='font-size: 12px; font-weight: 600; color: #0ea5e9; margin-bottom: 6px;'>📑 CITED PAPER (to check against)</div>", unsafe_allow_html=True)
        cited_paper_file = st.file_uploader("Upload Cited Reference Paper", type=["pdf"], label_visibility="collapsed")

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    
    col_action1, col_action2, _ = st.columns([1, 1, 2])
    with col_action1:
        verify_btn = st.button("Verify Citations", use_container_width=True)
    with col_action2:
        demo_btn = st.button("Try Demo", key="demo_btn", use_container_width=True)

    if verify_btn:
        if not main_paper_file or not cited_paper_file:
            st.warning("Please upload both a main paper and a cited reference paper to proceed.")
        else:
            run_verification(main_paper_file, cited_paper_file)

    if demo_btn:
        demo_data = demo_result()
        st.session_state.results = demo_data["results"]
        st.session_state.stats = demo_data["stats"]
        st.rerun()

    # Under the Hood Tech stack detailed cards
    st.markdown("""
    <div style="margin-top: 35px; border-top: 1px solid #e2e8f0; padding-top: 25px;">
        <h3 style="margin: 0 0 16px 0; font-size: 14px; font-weight: 700; color: #1e293b;">🔬 Under the Hood — RAG Pipeline</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
            <div style="background: #f8fafc; border-radius: 10px; padding: 12px 14px; border: 1px solid #e2e8f0;">
                <div style="font-size: 18px; margin-bottom: 6px;">📄</div>
                <div style="font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">PDF Extraction</div>
                <div style="font-size: 11.5px; color: #64748b; line-height: 1.4;">pdfplumber + PyMuPDF for robust, fallback-safe text extraction from research PDFs</div>
            </div>
            <div style="background: #f8fafc; border-radius: 10px; padding: 12px 14px; border: 1px solid #e2e8f0;">
                <div style="font-size: 18px; margin-bottom: 6px;">✂️</div>
                <div style="font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">Text Chunking</div>
                <div style="font-size: 11.5px; color: #64748b; line-height: 1.4;">Recursive splitting with 500-token chunk sizes and 80-token overlap boundaries</div>
            </div>
            <div style="background: #f8fafc; border-radius: 10px; padding: 12px 14px; border: 1px solid #e2e8f0;">
                <div style="font-size: 18px; margin-bottom: 6px;">🧠</div>
                <div style="font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">Embeddings</div>
                <div style="font-size: 11.5px; color: #64748b; line-height: 1.4;">Sentence Transformers (all-MiniLM-L6-v2) for local, fast normalized embeddings</div>
            </div>
            <div style="background: #f8fafc; border-radius: 10px; padding: 12px 14px; border: 1px solid #e2e8f0;">
                <div style="font-size: 18px; margin-bottom: 6px;">⚡</div>
                <div style="font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">FAISS Search</div>
                <div style="font-size: 11.5px; color: #64748b; line-height: 1.4;">Inner Product indices for rapid Approximate Nearest Neighbor semantic lookup</div>
            </div>
            <div style="background: #f8fafc; border-radius: 10px; padding: 12px 14px; border: 1px solid #e2e8f0;">
                <div style="font-size: 18px; margin-bottom: 6px;">🗄️</div>
                <div style="font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">ChromaDB Store</div>
                <div style="font-size: 11.5px; color: #64748b; line-height: 1.4;">Metadata-rich store querying cosine distances for vector cross-validation</div>
            </div>
            <div style="background: #f8fafc; border-radius: 10px; padding: 12px 14px; border: 1px solid #e2e8f0;">
                <div style="font-size: 18px; margin-bottom: 6px;">🔗</div>
                <div style="font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">LangChain RAG</div>
                <div style="font-size: 11.5px; color: #64748b; line-height: 1.4;">Multi-store retriever fusion step compiling the ultimate evidence base</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render results view
else:
    results = st.session_state.results
    stats = st.session_state.stats

    total_claims = stats["total_claims_verified"]
    supported = stats["supported"]
    contradicted = stats["contradicted"]
    insufficient = stats["insufficient"]
    overall_integrity = stats["overall_integrity"]

    integrity_colors = {
        "HIGH": {"bg": "#e8f5e9", "text": "#1b5e20", "border": "#4caf50"},
        "MEDIUM": {"bg": "#fff8e1", "text": "#e65100", "border": "#ff9800"},
        "LOW": {"bg": "#fce4ec", "text": "#880e4f", "border": "#e91e63"},
    }
    ic = integrity_colors.get(overall_integrity, {"bg": "#f1f5f9", "text": "#475569", "border": "#cbd5e1"})

    # Render dashboard metrics cards
    metrics_html = f"""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; margin-bottom: 16px;">
            <span style="font-size: 18px; margin-right: 8px;">📊</span>
            <h3 style="margin: 0; font-size: 15px; font-weight: 700; color: #1e293b; display: inline-block;">Verification Report</h3>
            <span style="margin-left: auto; padding: 4px 14px; border-radius: 20px; background: {ic['bg']}; color: {ic['text']}; font-size: 11px; font-weight: 700; border: 1px solid {ic['border']}80;">
                {overall_integrity} INTEGRITY
            </span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
            <div style="background: #fff; border-radius: 10px; padding: 12px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="font-size: 16px; margin-bottom: 2px;">🔍</div>
                <div style="font-size: 20px; font-weight: 700; color: #1e293b;">{total_claims}</div>
                <div style="font-size: 10px; color: #94a3b8; font-weight: 600;">Verified</div>
            </div>
            <div style="background: #fff; border-radius: 10px; padding: 12px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="font-size: 16px; margin-bottom: 2px;">✅</div>
                <div style="font-size: 20px; font-weight: 700; color: #1e293b;">{supported}</div>
                <div style="font-size: 10px; color: #94a3b8; font-weight: 600;">Supported</div>
            </div>
            <div style="background: #fff; border-radius: 10px; padding: 12px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="font-size: 16px; margin-bottom: 2px;">❌</div>
                <div style="font-size: 20px; font-weight: 700; color: #1e293b;">{contradicted}</div>
                <div style="font-size: 10px; color: #94a3b8; font-weight: 600;">Contradicted</div>
            </div>
            <div style="background: #fff; border-radius: 10px; padding: 12px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="font-size: 16px; margin-bottom: 2px;">⚠️</div>
                <div style="font-size: 20px; font-weight: 700; color: #1e293b;">{insufficient}</div>
                <div style="font-size: 10px; color: #94a3b8; font-weight: 600;">Insufficient</div>
            </div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

    # Charts section
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        pie_labels = []
        pie_values = []
        pie_colors = []
        if supported > 0:
            pie_labels.append("Supported")
            pie_values.append(supported)
            pie_colors.append("#4caf50")
        if contradicted > 0:
            pie_labels.append("Contradicted")
            pie_values.append(contradicted)
            pie_colors.append("#e91e63")
        if insufficient > 0:
            pie_labels.append("Insufficient")
            pie_values.append(insufficient)
            pie_colors.append("#ff9800")

        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=0.4,
            marker_colors=pie_colors,
            textinfo='percent',
            textfont=dict(size=11, family="Inter"),
            showlegend=True
        )])
        fig_pie.update_layout(
            title=dict(text="Verdict Distribution", font=dict(size=13, family="Inter", color="#475569")),
            margin=dict(l=5, r=5, t=35, b=5),
            height=180,
            legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    with col_chart2:
        metrics_names = ["Halluc. Risk", "Similarity", "Confidence", "Faithfulness"]
        metrics_vals = [
            round(stats["hallucination_risk"] * 100),
            round(stats["avg_similarity_score"] * 100),
            round(stats["avg_confidence"] * 100),
            round(stats["faithfulness_score"] * 100)
        ]
        
        fig_bar = go.Figure(go.Bar(
            x=metrics_vals,
            y=metrics_names,
            orientation='h',
            marker=dict(color='#6366f1', line=dict(width=0)),
            text=[f" {v}%" for v in metrics_vals],
            textposition='outside',
            textfont=dict(size=10, family="Inter")
        ))
        fig_bar.update_layout(
            title=dict(text="Quality Metrics (%)", font=dict(size=13, family="Inter", color="#475569")),
            margin=dict(l=5, r=25, t=35, b=5),
            height=180,
            xaxis=dict(range=[0, 115], showgrid=True, gridcolor='#f1f5f9', tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9, family="Inter")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # Filter selections and new verification button
    col_filter, col_reset = st.columns([3, 1])
    with col_filter:
        verdict_filter = st.radio(
            "Filter Results by Verdict",
            options=["ALL", "SUPPORTED", "CONTRADICTED", "INSUFFICIENT"],
            horizontal=True,
            label_visibility="collapsed"
        )
    with col_reset:
        new_verify = st.button("New Verification", use_container_width=True)
        if new_verify:
            st.session_state.results = None
            st.session_state.stats = None
            st.rerun()

    # Filter the results list
    filtered_results = results
    if verdict_filter != "ALL":
        filtered_results = [r for r in results if r["verdict"] == verdict_filter]

    # Render Result Cards
    st.write("---")
    if not filtered_results:
        st.markdown(f"<div style='text-align: center; padding: 30px; color: #94a3b8; font-size: 14px;'>No claims found matching verdict: <strong>{verdict_filter}</strong></div>", unsafe_allow_html=True)
    else:
        for r in filtered_results:
            verdict = r["verdict"]
            citation = r["citation_key"]
            claim = r["claim"]
            similarity = round(r["similarity_score"] * 100)
            confidence = round(r["confidence"] * 100)
            explanation = r["explanation"]
            evidence = r["evidence"]

            # Set styling parameters based on verdict
            colors = {
                "SUPPORTED": {"bg": "#e8f5e9", "text": "#1b5e20", "border": "#4caf50", "icon": "✓"},
                "CONTRADICTED": {"bg": "#fce4ec", "text": "#880e4f", "border": "#e91e63", "icon": "✗"},
                "INSUFFICIENT": {"bg": "#fff8e1", "text": "#e65100", "border": "#ff9800", "icon": "⚠"},
            }
            c = colors.get(verdict, colors["INSUFFICIENT"])

            # Clean and escape texts for safe HTML rendering
            safe_claim = claim.replace('"', '&quot;').replace("'", "&#39;")
            safe_explanation = explanation.replace('"', '&quot;').replace("'", "&#39;")
            
            # Format evidence block if it exists
            evidence_section = ""
            if evidence:
                safe_evidence = evidence.replace('"', '&quot;').replace("'", "&#39;")
                evidence_section = f"""
                <div style='background: #f8fafc; border-radius: 8px; padding: 10px 14px; border-left: 3px solid #94a3b8; margin-top: 8px;'>
                    <div style='font-size: 10px; font-weight: 700; color: #94a3b8; margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.5px;'>Key Evidence from Cited Paper</div>
                    <p style='margin: 0; font-size: 12.5px; color: #374151; line-height: 1.5; font-style: italic;'>"{safe_evidence}"</p>
                </div>
                """

            card_html = f"""
            <div style="border: 1px solid {c['border']}30; border-left: 5px solid {c['border']}; border-radius: 10px; background: white; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); padding: 14px 16px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;">
                    <div style="flex: 1;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                            <span style="font-size: 11px; font-weight: 700; color: #475569; background: #f1f5f9; padding: 2px 8px; border-radius: 4px;">
                                {citation}
                            </span>
                            <span style="background: {c['bg']}; color: {c['text']}; border: 1px solid {c['border']}80; border-radius: 20px; padding: 2px 10px; font-size: 11px; font-weight: 700; display: inline-flex; align-items: center; gap: 4px;">
                                {c['icon']} {verdict}
                            </span>
                        </div>
                        <p style="margin: 0; font-size: 14px; color: #1e293b; line-height: 1.5; font-weight: 500;">
                            {safe_claim}
                        </p>
                    </div>
                    <div style="text-align: right; min-width: 80px; flex-shrink: 0;">
                        <div style="font-size: 11px; color: #94a3b8; margin-bottom: 2px;">Similarity</div>
                        <div style="font-size: 13px; font-weight: 700; color: #334155;">{similarity}%</div>
                    </div>
                </div>
                
                <!-- Confidence Bar -->
                <div style="display: flex; align-items: center; gap: 8px; margin-top: 12px;">
                    <div style="flex: 1; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden;">
                        <div style="width: {confidence}%; height: 100%; background: {c['border']}; border-radius: 3px;"></div>
                    </div>
                    <span style="font-size: 11.5px; font-weight: 600; color: #64748b; min-width: 32px; text-align: right;">
                        {confidence}%
                    </span>
                </div>

                <!-- Custom native HTML details expander -->
                <details style="margin-top: 10px; border-top: 1px solid #f1f5f9; padding-top: 8px;">
                    <summary style="cursor: pointer; color: #6366f1; font-size: 12px; font-weight: 600; list-style: none;">
                        <span style="font-size: 10px; margin-right: 3px;">▼</span> Click to view explanation & evidence
                    </summary>
                    <div style="margin-top: 8px;">
                        <div style="margin-bottom: 6px;">
                            <div style="font-size: 10px; font-weight: 700; color: #94a3b8; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.5px;">AI Explanation</div>
                            <p style="margin: 0; font-size: 13px; color: #475569; line-height: 1.4;">{safe_explanation}</p>
                        </div>
                        {evidence_section}
                    </div>
                </details>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
