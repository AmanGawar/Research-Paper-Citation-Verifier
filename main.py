import os
import re
import json
import gc
import tempfile
import streamlit as st

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Citation Verifier",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Research Paper Citation Verifier")
st.caption("RAG-powered citation accuracy verification using FAISS & Sentence Transformers")

# ─── Sidebar Settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key (Optional)", type="password", help="Required for GPT verdict generation. If omitted, heuristic analysis is used.")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API Key set successfully")
    else:
        st.info("No API key set → using heuristic fallback mode")

st.markdown("---")

# ─── File Upload Section ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    main_pdf = st.file_uploader("📑 Upload Main Paper (PDF)", type=["pdf"], key="main_pdf")
with col2:
    cited_pdf = st.file_uploader("📚 Upload Cited Reference (PDF)", type=["pdf"], key="cited_pdf")

if main_pdf:
    st.caption(f"✓ Main paper: `{main_pdf.name}` ({main_pdf.size / 1024:.1f} KB)")
if cited_pdf:
    st.caption(f"✓ Cited paper: `{cited_pdf.name}` ({cited_pdf.size / 1024:.1f} KB)")

# ─── Helper Functions (Lazy Loaded) ───────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_pdf_text(uploaded_file) -> str:
    """Extract text safely from uploaded bytes using tempfiles."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        import pdfplumber
        with pdfplumber.open(tmp_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        pass

    if not text.strip():
        try:
            import fitz
            doc = fitz.open(tmp_path)
            text = "\n".join(page.get_text() for page in doc)
        except Exception:
            pass

    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    return text

def parse_citations(text: str) -> dict:
    cites = {}
    # Find numeric citations e.g. [1], [2]
    for m in re.findall(r'\[(\d+)\]', text):
        cites.setdefault(f"[{m}]", [])
    
    # Find Author-Year citations e.g. (Smith, 2020)
    for m in re.findall(r'\(([A-Z][a-zA-Z]+(?:\s+et\s+al\.)?,\s*\d{4})\)', text):
        cites.setdefault(f"({m})", [])

    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        for key in list(cites.keys()):
            bare_key = key.strip("[]()")
            if bare_key in sent or key in sent:
                if len(sent.strip()) > 20:
                    cites[key].append(sent.strip())

    return {k: v for k, v in cites.items() if v}

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list:
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 30:
            chunks.append(chunk)
    return chunks

def get_verdict(claim: str, evidence: list, citation_key: str) -> dict:
    open_key = os.getenv("OPENAI_API_KEY")
    if open_key and evidence:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=open_key)
            ctx = "\n\n".join(f"[{i+1}] {e}" for i, e in enumerate(evidence[:3]))
            prompt = f"""Verify if the CLAIM is supported by EVIDENCE.
CLAIM ({citation_key}): {claim}
EVIDENCE: {ctx}

Respond strictly in JSON format:
{{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT",
  "confidence": <float 0.0-1.0>,
  "explanation": "<one sentence>",
  "key_evidence": "<quote from evidence>"
}}"""
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250
            )
            raw = re.sub(r'^```json\s*|\s*```$', '', res.choices[0].message.content.strip(), flags=re.MULTILINE).strip()
            return json.loads(raw)
        except Exception:
            pass

    # Heuristic Fallback
    if not evidence:
        return {"verdict": "INSUFFICIENT", "confidence": 0.0, "explanation": "No matching evidence found in cited paper.", "key_evidence": ""}

    claim_words = set(re.findall(r'\b\w{4,}\b', claim.lower()))
    if not claim_words:
        return {"verdict": "INSUFFICIENT", "confidence": 0.0, "explanation": "Claim too short to analyze.", "key_evidence": ""}

    best_score, best_chunk = max([(len(claim_words & set(re.findall(r'\b\w{4,}\b', e.lower()))), e) for e in evidence])
    confidence = min(best_score / max(len(claim_words), 1), 1.0)
    verdict = "SUPPORTED" if confidence >= 0.35 else "INSUFFICIENT"

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "explanation": "Keyword and semantic similarity match.",
        "key_evidence": best_chunk[:300]
    }

# ─── Execution Trigger ────────────────────────────────────────────────────────
can_run = main_pdf is not None and cited_pdf is not None

if st.button("🚀 Start Verification", type="primary", disabled=not can_run):
    with st.spinner("Processing documents & running RAG pipeline..."):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Text Extraction
            status_text.text("📖 Extracting text from PDFs...")
            progress_bar.progress(10)
            main_text = extract_pdf_text(main_pdf)
            cited_text = extract_pdf_text(cited_pdf)

            if not main_text or not cited_text:
                st.error("❌ Failed to extract text from one or both PDFs. Ensure they are text-readable (not scanned images).")
                st.stop()

            # Step 2: Parse Citations
            status_text.text("🔍 Parsing citation keys and claims...")
            progress_bar.progress(25)
            citations = parse_citations(main_text)

            if not citations:
                sentences = re.split(r'(?<=[.!?])\s+', main_text)[:20]
                citations = {"[General Claims]": [s for s in sentences if len(s) > 40]}

            # Step 3: Chunking & Indexing
            status_text.text("⚡ Chunking text and building FAISS vector index...")
            progress_bar.progress(40)
            chunks = chunk_text(cited_text)

            if not chunks:
                st.error("❌ Cited paper contains no indexable text chunks.")
                st.stop()

            embedder = load_embedder()
            chunk_embeddings = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

            import numpy as np
            import faiss

            dim = chunk_embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(chunk_embeddings.astype("float32"))

            # Step 4: Claim Verification
            status_text.text("🤖 Matching claims against evidence...")
            progress_bar.progress(60)

            results = []
            citation_items = list(citations.items())[:15]

            for idx, (cite_key, sentences) in enumerate(citation_items):
                # Select top 2 longest sentences as claims
                claims = sorted(sentences, key=len, reverse=True)[:2]

                for claim in claims:
                    q_emb = embedder.encode([claim], show_progress_bar=False, normalize_embeddings=True).astype("float32")
                    scores, indices = faiss_index.search(q_emb, k=3)

                    matched_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
                    top_score = float(scores[0][0]) if len(scores[0]) > 0 else 0.0

                    verdict_info = get_verdict(claim, matched_chunks, cite_key)

                    results.append({
                        "claim": claim[:400],
                        "citation_key": cite_key,
                        "verdict": verdict_info.get("verdict", "INSUFFICIENT"),
                        "confidence": verdict_info.get("confidence", 0.0),
                        "evidence": verdict_info.get("key_evidence", "")[:300],
                        "explanation": verdict_info.get("explanation", "")[:250],
                        "similarity_score": round(top_score, 4)
                    })

                progress_bar.progress(60 + int((idx + 1) / len(citation_items) * 35))

            progress_bar.progress(100)
            status_text.text("✅ Verification complete!")

            # Memory Cleanup
            gc.collect()

            # Step 5: Display Summary & Metrics
            st.markdown("---")
            st.header("📊 Verification Metrics")

            total = len(results)
            supported = sum(1 for r in results if r["verdict"] == "SUPPORTED")
            contradicted = sum(1 for r in results if r["verdict"] == "CONTRADICTED")
            insufficient = total - supported - contradicted
            faithfulness = (supported / total) if total > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Claims Evaluated", total)
            c2.metric("🟢 Supported", supported)
            c3.metric("🔴 Contradicted", contradicted)
            c4.metric("Integrity Rating", "HIGH" if faithfulness >= 0.7 else "MEDIUM" if faithfulness >= 0.4 else "LOW")

            st.markdown("---")
            st.header("📋 Citation Breakdown")

            for i, res in enumerate(results, 1):
                icon = "🟢" if res["verdict"] == "SUPPORTED" else "🔴" if res["verdict"] == "CONTRADICTED" else "🟡"
                with st.expander(f"Claim {i} {icon} [{res['citation_key']}] - {res['verdict']}", expanded=(i <= 3)):
                    st.markdown(f"**Claim:** {res['claim']}")
                    st.markdown(f"**Verdict:** `{res['verdict']}` | **Confidence:** `{res['confidence']:.1%}` | **Semantic Match:** `{res['similarity_score']:.3f}`")
                    st.markdown(f"**Explanation:** {res['explanation']}")
                    if res["evidence"]:
                        st.info(f"**Cited Evidence:** {res['evidence']}")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)
