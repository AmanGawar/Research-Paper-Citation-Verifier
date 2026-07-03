import os
import re
import json
import uuid
import logging
import tempfile
import streamlit as st
from typing import Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Citation Verifier", page_icon="📄", layout="wide")

if "jobs" not in st.session_state:
    st.session_state.jobs = {}
if "active_job" not in st.session_state:
    st.session_state.active_job = None
if "processing" not in st.session_state:
    st.session_state.processing = False

st.title("📄 Research Paper Citation Verifier")
st.markdown("Upload papers and verify citations using RAG + Vector Search")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key (optional)", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API key configured")
    else:
        st.info("Using heuristic mode (no LLM)")

st.markdown("---")

main_file = st.file_uploader("📄 Main Paper", type=["pdf"], key="main_upload")
cited_file = st.file_uploader("📚 Cited Paper", type=["pdf"], key="cited_upload")

if main_file:
    st.success(f"✅ Main paper loaded: {main_file.name} ({main_file.size/1024:.1f} KB)")
if cited_file:
    st.success(f"✅ Cited paper loaded: {cited_file.name} ({cited_file.size/1024:.1f} KB)")

can_process = main_file is not None and cited_file is not None and not st.session_state.processing

if st.button("🚀 Verify Citations", disabled=not can_process, type="primary"):
    st.session_state.processing = True
    job_id = str(uuid.uuid4())[:8]
    st.session_state.active_job = job_id
    
    with st.spinner("Processing PDFs..."):
        try:
            main_bytes = main_file.read()
            cited_bytes = cited_file.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f1:
                f1.write(main_bytes)
                main_path = f1.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f2:
                f2.write(cited_bytes)
                cited_path = f2.name
            
            st.session_state.jobs[job_id] = {
                "status": "processing",
                "main_path": main_path,
                "cited_path": cited_path,
                "progress": 0,
                "message": "Starting..."
            }
            
            process_job(job_id, main_path, cited_path)
            
        except Exception as e:
            st.error(f"❌ Upload error: {str(e)}")
            st.session_state.processing = False

if st.session_state.active_job:
    job_id = st.session_state.active_job
    if job_id in st.session_state.jobs:
        job = st.session_state.jobs[job_id]
        
        if job["status"] == "processing":
            st.info(f"⏳ {job['message']}")
            st.progress(job["progress"] / 100)
            
        elif job["status"] == "done":
            st.session_state.processing = False
            st.success("✅ Verification Complete!")
            
            stats = job.get("stats", {})
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Claims", stats.get("total_claims_verified", 0))
                col2.metric("Supported", stats.get("supported", 0))
                col3.metric("Contradicted", stats.get("contradicted", 0))
                col4.metric("Integrity", stats.get("overall_integrity", "N/A"))
            
            results = job.get("results", [])
            if results:
                st.subheader("📊 Results")
                for i, r in enumerate(results, 1):
                    icon = {"SUPPORTED": "🟢", "CONTRADICTED": "🔴", "INSUFFICIENT": "🟡"}.get(r["verdict"], "⚪")
                    with st.expander(f"{icon} Claim {i}: {r['citation_key']} - {r['verdict']}", expanded=i<=3):
                        st.markdown(f"**Claim:** {r['claim'][:300]}...")
                        st.metric("Confidence", f"{r['confidence']:.1%}")
                        st.markdown(f"**Explanation:** {r['explanation']}")
                        st.text_area("Evidence", r['evidence'], height=100, key=f"ev_{i}")
            
            if st.button("🔄 Verify Another"):
                st.session_state.active_job = None
                st.session_state.processing = False
                st.rerun()
                
        elif job["status"] == "error":
            st.session_state.processing = False
            st.error(f"❌ Error: {job.get('error', 'Unknown')}")
            if st.button("Try Again"):
                st.session_state.active_job = None
                st.rerun()

def process_job(job_id, main_path, cited_path):
    try:
        st.session_state.jobs[job_id]["progress"] = 10
        st.session_state.jobs[job_id]["message"] = "Extracting text..."
        
        main_text = extract_pdf_text(main_path)
        cited_text = extract_pdf_text(cited_path)
        
        st.session_state.jobs[job_id]["progress"] = 30
        st.session_state.jobs[job_id]["message"] = "Analyzing citations..."
        
        citations = extract_citations(main_text)
        if not citations:
            citations = {"[General]": re.split(r'[.!?]\s+', main_text)[:20]}
        
        st.session_state.jobs[job_id]["progress"] = 50
        st.session_state.jobs[job_id]["message"] = "Building vector index..."
        
        chunks = simple_chunk(cited_text)
        embeddings = get_embeddings(chunks)
        
        st.session_state.jobs[job_id]["progress"] = 70
        st.session_state.jobs[job_id]["message"] = "Verifying claims..."
        
        results = []
        for cite_key, sentences in list(citations.items())[:10]:
            claims = extract_key_claims(sentences)
            for claim in claims[:2]:
                evidence = semantic_search(claim, chunks, embeddings)
                verdict = verify_claim(claim, evidence)
                results.append({
                    "claim": claim[:300],
                    "citation_key": cite_key,
                    "verdict": verdict["verdict"],
                    "confidence": verdict["confidence"],
                    "explanation": verdict["explanation"],
                    "evidence": verdict["evidence"][:200],
                    "similarity_score": verdict.get("similarity", 0.0)
                })
        
        st.session_state.jobs[job_id]["progress"] = 90
        stats = compute_statistics(results)
        
        st.session_state.jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "message": "Complete",
            "results": results,
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        st.session_state.jobs[job_id].update({
            "status": "error",
            "error": str(e)
        })
    finally:
        for p in [main_path, cited_path]:
            try:
                os.unlink(p)
            except:
                pass

def extract_pdf_text(path):
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except:
        import fitz
        return "\n".join(p.get_text() for p in fitz.open(path))

def extract_citations(text):
    cites = {}
    for m in re.findall(r'\[(\d+)\]', text):
        cites.setdefault(f"[{m}]", [])
    for sent in re.split(r'[.!?]\s+', text):
        for key in cites.keys():
            if key in sent:
                cites[key].append(sent)
    return {k:v for k,v in cites.items() if v}

def extract_key_claims(sentences):
    keywords = ['show', 'demonstrate', 'find', 'prove', 'suggest', 'indicate', 'confirm']
    scored = [(sum(kw in s.lower() for kw in keywords), s) for s in sentences if len(s) > 40]
    scored.sort(reverse=True)
    return [s for _, s in scored[:2]]

def simple_chunk(text, size=500):
    words = text.split()
    return [' '.join(words[i:i+size]) for i in range(0, len(words), size-50)]

@st.cache_resource
def get_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    model = get_sentence_model()
    return model.encode(texts, normalize_embeddings=True)

def semantic_search(query, chunks, embeddings, top_k=3):
    model = get_sentence_model()
    q_emb = model.encode([query], normalize_embeddings=True)
    import numpy as np
    scores = np.dot(embeddings, q_emb.T).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_idx]

def verify_claim(claim, evidence):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and evidence:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = f"Does this evidence support the claim?\nClaim: {claim}\nEvidence: {evidence[0][:500]}\nRespond with JSON: {{\"verdict\":\"SUPPORTED|CONTRADICTED|INSUFFICIENT\",\"confidence\":0-1,\"explanation\":\"...\",\"evidence\":\"...\"}}"
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            return json.loads(re.sub(r'```json|```', '', resp.choices[0].message.content).strip())
        except:
            pass
    
    if not evidence:
        return {"verdict": "INSUFFICIENT", "confidence": 0.0, "explanation": "No evidence", "evidence": ""}
    
    claim_words = set(claim.lower().split())
    ev_words = set(evidence[0].lower().split())
    overlap = len(claim_words & ev_words) / max(len(claim_words), 1)
    
    return {
        "verdict": "SUPPORTED" if overlap > 0.3 else "INSUFFICIENT",
        "confidence": round(overlap, 3),
        "explanation": "Keyword overlap analysis",
        "evidence": evidence[0][:200],
        "similarity": overlap
    }

def compute_statistics(results):
    if not results:
        return {}
    total = len(results)
    supported = sum(1 for r in results if r["verdict"] == "SUPPORTED")
    contradicted = sum(1 for r in results if r["verdict"] == "CONTRADICTED")
    return {
        "total_claims_verified": total,
        "supported": supported,
        "contradicted": contradicted,
        "insufficient": total - supported - contradicted,
        "faithfulness_score": round(supported/total, 2),
        "hallucination_risk": round(contradicted/total, 2),
        "avg_confidence": round(sum(r["confidence"] for r in results)/total, 2),
        "avg_similarity_score": round(sum(r.get("similarity_score", 0) for r in results)/total, 2),
        "overall_integrity": "HIGH" if supported/total > 0.7 else "MEDIUM" if supported/total > 0.4 else "LOW"
    }
