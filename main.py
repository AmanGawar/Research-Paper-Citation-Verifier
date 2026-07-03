import os
import re
import json
import tempfile
import streamlit as st

st.set_page_config(page_title="Citation Verifier", page_icon="📄", layout="wide")

st.title("📄 Research Paper Citation Verifier")
st.markdown("Upload two PDFs to verify citation accuracy using FAISS vector search")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key (Optional)", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API Key set")
    else:
        st.info("ℹ️ Using keyword-based analysis")

st.markdown("---")

main_pdf = st.file_uploader("📄 Main Paper", type=["pdf"])
cited_pdf = st.file_uploader("📚 Cited Reference", type=["pdf"])

if main_pdf and cited_pdf:
    st.success(f"✅ Files loaded: {main_pdf.name} & {cited_pdf.name}")

if st.button("🚀 Verify Citations", disabled=not (main_pdf and cited_pdf), type="primary"):
    
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Step 1: Extract text
        status.text("📖 Extracting text from PDFs...")
        progress.progress(20)
        
        main_text = extract_text(main_pdf)
        cited_text = extract_text(cited_pdf)
        
        if not main_text or not cited_text:
            st.error("❌ Could not extract text from PDFs")
            st.stop()
        
        # Step 2: Parse citations
        status.text("🔍 Parsing citations...")
        progress.progress(40)
        
        citations = find_citations(main_text)
        if not citations:
            citations = {"[General]": main_text.split(". ")[:10]}
        
        # Step 3: Build index
        status.text("⚡ Building vector index...")
        progress.progress(60)
        
        chunks = chunk_text(cited_text)
        embeddings, chunks = build_index(chunks)
        
        # Step 4: Verify claims
        status.text("🤖 Verifying claims...")
        progress.progress(80)
        
        results = []
        for cite_key, sentences in list(citations.items())[:10]:
            for claim in sentences[:2]:
                if len(claim) < 30:
                    continue
                evidence = search_evidence(claim, chunks, embeddings)
                verdict = verify(claim, evidence)
                results.append({
                    "citation": cite_key,
                    "claim": claim[:200],
                    "verdict": verdict["verdict"],
                    "confidence": verdict["confidence"],
                    "evidence": verdict["evidence"][:200]
                })
        
        progress.progress(100)
        status.text("✅ Complete!")
        
        # Display results
        st.markdown("---")
        st.header("📊 Results")
        
        total = len(results)
        supported = sum(1 for r in results if r["verdict"] == "SUPPORTED")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Claims", total)
        col2.metric("Supported", supported)
        col3.metric("Accuracy", f"{supported/total*100:.0f}%" if total > 0 else "N/A")
        
        st.markdown("---")
        
        for i, r in enumerate(results, 1):
            icon = "🟢" if r["verdict"] == "SUPPORTED" else "🟡"
            with st.expander(f"{icon} Claim {i} [{r['citation']}]", expanded=i<=3):
                st.markdown(f"**Claim:** {r['claim']}")
                st.metric("Confidence", f"{r['confidence']:.0%}")
                st.text_area("Evidence", r["evidence"], height=80, key=f"ev{i}")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

# Helper functions
def extract_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except:
        pass
    
    if not text:
        try:
            import fitz
            text = "\n".join(p.get_text() for p in fitz.open(path))
        except:
            pass
    
    try:
        os.unlink(path)
    except:
        pass
    
    return text

def find_citations(text):
    cites = {}
    for m in re.findall(r'\[(\d+)\]', text):
        cites.setdefault(f"[{m}]", [])
    
    for sent in text.split(". "):
        for key in cites.keys():
            if key in sent and len(sent) > 30:
                cites[key].append(sent)
    
    return {k: v for k, v in cites.items() if v}

def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-50)]

@st.cache_resource
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

def build_index(chunks):
    if not chunks:
        return None, []
    model = get_model()
    embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    return embeddings, chunks

def search_evidence(query, chunks, embeddings, top_k=3):
    if embeddings is None or len(chunks) == 0:
        return []
    model = get_model()
    q_emb = model.encode([query], show_progress_bar=False, normalize_embeddings=True)
    
    import numpy as np
    scores = np.dot(embeddings, q_emb.T).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]
    
    return [chunks[i] for i in top_idx if i < len(chunks)]

def verify(claim, evidence):
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key and evidence:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            ctx = " ".join(evidence[:2])
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Does this evidence support the claim?\nClaim: {claim}\nEvidence: {ctx}\nRespond with JSON: {{\"verdict\":\"SUPPORTED\" or \"INSUFFICIENT\",\"confidence\":0-1,\"evidence\":\"quote\"}}"}],
                temperature=0,
                max_tokens=150
            )
            raw = re.sub(r'```json|```', '', resp.choices[0].message.content).strip()
            return json.loads(raw)
        except:
            pass
    
    if not evidence:
        return {"verdict": "INSUFFICIENT", "confidence": 0.0, "evidence": ""}
    
    claim_words = set(claim.lower().split())
    ev_words = set(evidence[0].lower().split())
    overlap = len(claim_words & ev_words) / max(len(claim_words), 1)
    
    return {
        "verdict": "SUPPORTED" if overlap > 0.25 else "INSUFFICIENT",
        "confidence": overlap,
        "evidence": evidence[0][:200]
    }
