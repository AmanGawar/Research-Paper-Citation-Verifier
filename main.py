import os
import re
import json
import uuid
import logging
import tempfile
import threading
import streamlit as st
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Research Paper Citation Verifier", page_icon="📄", layout="wide")

for key, default in [("jobs", {}), ("active_job", None), ("running", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("📄 Research Paper Citation Verifier")
st.success("✅ App is live! Upload PDFs below (max 10MB each)")

with st.sidebar:
    st.header("🔑 Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Optional - enables LLM-powered verification")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API Key set")
    else:
        st.info("ℹ️ No API key - using keyword analysis mode")
    
    st.markdown("---")
    st.caption("⚠️ Keep PDFs under 10MB for best performance")
    st.caption("🔒 Files are processed in memory and deleted after analysis")

col1, col2 = st.columns(2)
with col1:
    main_paper = st.file_uploader("📑 Main Paper (PDF)", type=["pdf"], key="main", help="The paper you want to verify")
with col2:
    cited_paper = st.file_uploader("📚 Cited Reference (PDF)", type=["pdf"], key="cited", help="The paper being cited")

# File size validation
if main_paper and main_paper.size > 10 * 1024 * 1024:
    st.error("❌ Main paper exceeds 10MB. Please use a smaller file.")
    main_paper = None
    
if cited_paper and cited_paper.size > 10 * 1024 * 1024:
    st.error("❌ Cited paper exceeds 10MB. Please use a smaller file.")
    cited_paper = None

if st.button("🚀 Start Verification", type="primary", disabled=st.session_state.running or not (main_paper and cited_paper)):
    try:
        job_id = str(uuid.uuid4())
        st.session_state.active_job = job_id
        st.session_state.running = True
        
        # Read files immediately to avoid async issues
        main_bytes = main_paper.read()
        cited_bytes = cited_paper.read()
        
        # Save to temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t1:
            t1.write(main_bytes)
            main_path = t1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t2:
            t2.write(cited_bytes)
            cited_path = t2.name
        
        st.info("📤 Files uploaded successfully. Processing...")
        
        # Start background processing
        thread = threading.Thread(target=lambda: run_verification_job(job_id, main_path, cited_path), daemon=True)
        thread.start()
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}")
        st.session_state.running = False

if st.session_state.active_job and st.session_state.active_job in st.session_state.jobs:
    job = st.session_state.jobs[st.session_state.active_job]
    
    progress_container = st.container()
    with progress_container:
        st.progress(job["progress"] / 100)
        st.markdown(f"**Status:** `{job['status'].upper()}` | **Progress:** `{job['progress']}%`")
        st.caption(job["message"])
    
    if job["status"] == "processing":
        st.rerun()
    
    if job["status"] == "done":
        st.session_state.running = False
        st.success("✅ Verification Complete!")
        
        if job.get("stats"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Claims", job["stats"]["total_claims_verified"])
            c2.metric("✅ Supported", job["stats"]["supported"])
            c3.metric("❌ Contradicted", job["stats"]["contradicted"])
            c4.metric("Integrity Score", job["stats"]["overall_integrity"])
            
        st.subheader("📋 Detailed Results")
        for i, res in enumerate(job["results"], 1):
            icon = "🟢" if res["verdict"]=="SUPPORTED" else "🔴" if res["verdict"]=="CONTRADICTED" else "🟡"
            with st.expander(f"Claim {i} [{res['citation_key']}] {icon} {res['verdict']}", expanded=i<=3):
                st.markdown(f"**Claim:** {res['claim'][:300]}...")
                st.markdown(f"**Confidence:** `{res['confidence']:.1%}` | **Similarity:** `{res['similarity_score']:.3f}`")
                st.markdown(f"**Explanation:** {res['explanation']}")
                with st.container(border=True):
                    st.caption("Evidence from cited paper:")
                    st.write(res['evidence'])
                
        if st.button("🔄 Verify Another Paper"):
            st.session_state.active_job = None
            st.session_state.running = False
            st.rerun()
                
    elif job["status"] == "error":
        st.session_state.running = False
        st.error(f"❌ Processing failed: {job.get('error', 'Unknown error')}")
        if st.button("🔄 Try Again"):
            st.session_state.active_job = None
            st.session_state.running = False
            st.rerun()

def run_verification_job(job_id: str, main_path: str, cited_path: str):
    st.session_state.jobs[job_id] = {"status": "processing", "progress": 5, "message": "Initializing..."}
    try:
        import numpy as np
        import faiss
        import chromadb
        from chromadb.config import Settings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS as LangFAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from sentence_transformers import SentenceTransformer
        
        st.session_state.jobs[job_id]["progress"] = 10
        st.session_state.jobs[job_id]["message"] = "Extracting PDF text..."
        main_text = extract_text(main_path)
        cited_text = extract_text(cited_path)
        
        if not main_text or not cited_text:
            raise ValueError("Could not extract text from one or both PDFs")
        
        st.session_state.jobs[job_id]["progress"] = 20
        st.session_state.jobs[job_id]["message"] = "Parsing citations..."
        citations = parse_citations(main_text)
        if not citations:
            citations = {"[ALL]": re.split(r'(?<=[.!?])\s+', main_text)[:30]}
            
        st.session_state.jobs[job_id]["progress"] = 30
        st.session_state.jobs[job_id]["message"] = "Chunking text..."
        chunks = chunk_text(cited_text)
        
        st.session_state.jobs[job_id]["message"] = "Loading embedding model..."
        embedder = load_embedder()
        embeddings = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        
        st.session_state.jobs[job_id]["progress"] = 40
        st.session_state.jobs[job_id]["message"] = "Building FAISS index..."
        faiss_idx = faiss.IndexFlatIP(embeddings.shape[1])
        faiss_idx.add(embeddings.astype('float32'))
        
        st.session_state.jobs[job_id]["progress"] = 50
        st.session_state.jobs[job_id]["message"] = "Building ChromaDB..."
        chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        col_name = f"c{job_id[:6]}"
        try:
            chroma_client.delete_collection(col_name)
        except:
            pass
        col = chroma_client.create_collection(name=col_name)
        col.add(embeddings=embeddings.tolist(), documents=chunks, ids=[f"c{i}" for i in range(len(chunks))])
        
        st.session_state.jobs[job_id]["progress"] = 60
        st.session_state.jobs[job_id]["message"] = "Building LangChain retriever..."
        lc_emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        lc_ret = LangFAISS.from_texts(chunks, lc_emb).as_retriever(search_kwargs={"k": 5})
        
        st.session_state.jobs[job_id]["progress"] = 70
        st.session_state.jobs[job_id]["message"] = "Verifying claims..."
        results = []
        items = list(citations.items())[:15]
        
        for i, (cite_key, sentences) in enumerate(items):
            claims = get_claims(sentences) or sentences[:1]
            for claim in claims[:2]:
                q_vec = embedder.encode([claim], show_progress_bar=False, normalize_embeddings=True).astype('float32')
                scores, indices = faiss_idx.search(q_vec, 5)
                faiss_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
                
                chroma_res = col.query(query_embeddings=embedder.encode([claim]).tolist(), n_results=3)
                all_evidence = list(dict.fromkeys(faiss_chunks + chroma_res['documents'][0]))[:4]
                
                try:
                    lc_docs = lc_ret.invoke(claim)
                    all_evidence = list(dict.fromkeys(all_evidence + [d.page_content for d in lc_docs]))[:4]
                except:
                    pass
                
                verdict = get_verdict(claim, all_evidence, cite_key)
                results.append({
                    "claim": claim[:400], "citation_key": cite_key,
                    "verdict": verdict.get("verdict", "INSUFFICIENT"),
                    "confidence": verdict.get("confidence", 0.0),
                    "evidence": verdict.get("key_evidence", "")[:300],
                    "explanation": verdict.get("explanation", "")[:250],
                    "similarity_score": round(float(scores[0][0]) if len(scores[0])>0 else 0.0, 4)
                })
            
            progress = 70 + int((i/len(items))*25)
            st.session_state.jobs[job_id]["progress"] = progress
            st.session_state.jobs[job_id]["message"] = f"Verified {i+1}/{len(items)} citations..."
            
        st.session_state.jobs[job_id]["progress"] = 95
        st.session_state.jobs[job_id]["message"] = "Computing statistics..."
        stats = compute_stats(results)
        
        st.session_state.jobs[job_id].update({
            "status": "done", 
            "progress": 100, 
            "message": "Complete!", 
            "results": results, 
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        st.session_state.jobs[job_id].update({
            "status": "error", 
            "progress": 0, 
            "message": "Failed", 
            "error": str(e)
        })
    finally:
        for p in [main_path, cited_path]:
            try:
                os.unlink(p)
            except:
                pass

def extract_text(path: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                return text
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
    
    try:
        import fitz
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error(f"PyMuPDF also failed: {e}")
        return ""

def parse_citations(text: str) -> dict:
    cites = {}
    for m in re.findall(r'\[(\d+(?:[,\s-]\d+)*)\]', text):
        for k in re.split(r'[,\s-]+', m):
            k = k.strip()
            if k:
                cites.setdefault(f"[{k}]", [])
    for m in re.findall(r'\(([A-Z][a-zA-Z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?)\)', text):
        cites.setdefault(f"({m})", [])
    for sent in re.split(r'(?<=[.!?])\s+', text):
        for k in list(cites.keys()):
            if k.strip('[]()') in sent or k in sent:
                cites[k].append(sent.strip())
    return {k:v for k,v in cites.items() if v}

def get_claims(sents: list) -> list:
    kw = ['show','find','found','report','suggest','indicate','confirm','propose','argue','claim','conclude','improve','increase','decrease','higher','lower','better']
    scored = [(sum(1 for w in kw if w in s.lower()), s) for s in sents if len(s)>30]
    scored.sort(reverse=True)
    return [s for _,s in scored[:3]]

def chunk_text(text: str) -> list:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_text(text)

@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_verdict(claim: str, evidence: list, cite_key: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not evidence:
        return heuristic_verdict(claim, evidence)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        ctx = "\n\n".join(f"[{i+1}] {e}" for i,e in enumerate(evidence[:3]))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Verify CLAIM vs EVIDENCE.\nCLAIM: {claim}\nEVIDENCE: {ctx}\nJSON only: {{\"verdict\":\"SUPPORTED\"|\"CONTRADICTED\"|\"INSUFFICIENT\",\"confidence\":0-1,\"explanation\":\"...\",\"key_evidence\":\"...\"}}"}],
            temperature=0.1, max_tokens=300
        )
        raw = re.sub(r'^```json\s*|\s*```$', '', resp.choices[0].message.content.strip()).strip()
        return json.loads(raw)
    except:
        return heuristic_verdict(claim, evidence)

def heuristic_verdict(claim: str, evidence: list) -> dict:
    if not evidence:
        return {"verdict":"INSUFFICIENT","confidence":0.0,"explanation":"No evidence found","key_evidence":""}
    c_words = set(re.findall(r'\b\w{4,}\b', claim.lower()))
    best_score, best_chunk = max([(len(c_words & set(re.findall(r'\b\w{4,}\b', e.lower()))), e) for e in evidence])
    conf = min(best_score/max(len(c_words),1), 1.0)
    v = "SUPPORTED" if conf>=0.35 else "INSUFFICIENT"
    return {"verdict":v,"confidence":round(conf,3),"explanation":"Keyword-based analysis","key_evidence":best_chunk[:300]}

def compute_stats(res: list) -> dict:
    if not res:
        return {}
    t = len(res)
    s = sum(1 for r in res if r["verdict"]=="SUPPORTED")
    c = sum(1 for r in res if r["verdict"]=="CONTRADICTED")
    f = s/t
    return {
        "total_claims_verified":t,"supported":s,"contradicted":c,"insufficient":t-s-c,
        "faithfulness_score":round(f,3),"hallucination_risk":round(c/t,3),
        "avg_confidence":round(sum(r["confidence"] for r in res)/t,3),
        "avg_similarity_score":round(sum(r["similarity_score"] for r in res)/t,3),
        "overall_integrity":"HIGH" if f>=0.7 else "MEDIUM" if f>=0.4 else "LOW"
    }
