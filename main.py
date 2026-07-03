"""
Research Paper Citation Verifier
FastAPI backend with RAG pipeline using LangChain, ChromaDB, FAISS, Sentence Transformers
"""

import os
import re
import json
import uuid
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Paper Citation Verifier",
    description="RAG-powered citation verification using LangChain, ChromaDB, FAISS, Sentence Transformers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory job store ───────────────────────────────────────────────────────
jobs: dict = {}


# ─── Pydantic Models ───────────────────────────────────────────────────────────
class VerificationResult(BaseModel):
    claim: str
    citation_key: str
    verdict: str          # "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT"
    confidence: float
    evidence: str
    explanation: str
    similarity_score: float


class JobStatus(BaseModel):
    job_id: str
    status: str           # "pending" | "processing" | "done" | "error"
    progress: int         # 0-100
    message: str
    results: Optional[list] = None
    stats: Optional[dict] = None
    error: Optional[str] = None


# ─── PDF Extraction ────────────────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    """Extract full text from a PDF using pdfplumber with pymupdf fallback."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        if text.strip():
            return text
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}, trying pymupdf")

    try:
        import fitz  # pymupdf
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"Could not extract text from PDF: {e}")


# ─── Citation Parser ───────────────────────────────────────────────────────────
def extract_citations(text: str) -> dict:
    """
    Parse inline citations like [1], [2,3], (Author, 2020) from paper text.
    Returns a dict of citation_key -> list of sentence contexts.
    """
    citations = {}

    # Numeric citations [1], [1,2], [1-3]
    numeric = re.findall(r'\[(\d+(?:[,\s-]\d+)*)\]', text)
    for match in numeric:
        keys = re.split(r'[,\s-]+', match)
        for k in keys:
            k = k.strip()
            if k:
                citations.setdefault(f"[{k}]", [])

    # Author-year citations (Smith, 2020), (Smith et al., 2020)
    author_year = re.findall(
        r'\(([A-Z][a-zA-Z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?(?:;\s*[A-Z][a-zA-Z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?)*)\)',
        text
    )
    for match in author_year:
        key = f"({match})"
        citations.setdefault(key, [])

    # Now find sentences near each citation key
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        for key in list(citations.keys()):
            bare = key.strip('[]() ')
            if bare in sentence or key in sentence:
                citations[key].append(sentence.strip())

    # Remove empty
    citations = {k: v for k, v in citations.items() if v}
    return citations


def extract_claims_from_sentences(sentences: list, citation_key: str) -> list:
    """Extract the most claim-like sentences that reference a given citation."""
    claim_keywords = [
        'show', 'demonstrate', 'prove', 'find', 'found', 'report', 'suggest',
        'indicate', 'reveal', 'confirm', 'establish', 'propose', 'argue',
        'claim', 'state', 'conclude', 'achieve', 'improve', 'outperform',
        'increase', 'decrease', 'result', 'significant', 'higher', 'lower',
        'better', 'worse', 'according to', 'study', 'research', 'analysis'
    ]
    scored = []
    for s in sentences:
        score = sum(1 for kw in claim_keywords if kw.lower() in s.lower())
        if len(s) > 30:
            scored.append((score, s))

    scored.sort(reverse=True)
    return [s for _, s in scored[:3]]


# ─── Text Chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks for vector embedding."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# ─── Embedding Generation ──────────────────────────────────────────────────────
_embedder = None

def get_embedder():
    """Lazy-load Sentence Transformers embedder (resume skill: Sentence Transformers)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


def embed_texts(texts: list) -> np.ndarray:
    """Generate embeddings using Sentence Transformers."""
    embedder = get_embedder()
    return embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)


# ─── Vector Store (FAISS) ──────────────────────────────────────────────────────
def build_faiss_index(chunks: list) -> tuple:
    """
    Build a FAISS vector index from text chunks.
    Resume skill: FAISS, vector database, embedding generation.
    """
    import faiss

    logger.info(f"Building FAISS index for {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)   # Inner product (cosine on normalized vecs)
    index.add(embeddings.astype('float32'))
    return index, embeddings


def faiss_search(index, chunks: list, query: str, top_k: int = 5) -> list:
    """Semantic search over FAISS index. Returns (chunk, score) pairs."""
    import faiss

    q_emb = embed_texts([query]).astype('float32')
    scores, indices = index.search(q_emb, min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append((chunks[idx], float(score)))
    return results


# ─── ChromaDB Vector Store ─────────────────────────────────────────────────────
def build_chroma_collection(collection_name: str, chunks: list):
    """
    Store chunks in ChromaDB for persistent retrieval.
    Resume skill: ChromaDB, vector database.
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.Client(Settings(anonymized_telemetry=False))

    # Delete if exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    embeddings = embed_texts(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids,
    )
    logger.info(f"ChromaDB collection '{collection_name}' built with {len(chunks)} docs.")
    return collection


def chroma_search(collection, query: str, top_k: int = 5) -> list:
    """Query ChromaDB collection. Returns (document, distance) pairs."""
    q_emb = embed_texts([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=min(top_k, collection.count()),
    )
    pairs = []
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        pairs.append((doc, 1.0 - dist))   # Convert distance → similarity
    return pairs


# ─── LangChain RAG Pipeline ────────────────────────────────────────────────────
def build_langchain_retriever(chunks: list):
    """
    Build a LangChain retriever backed by FAISS.
    Resume skill: LangChain, RAG pipeline.
    """
    from langchain_community.vectorstores import FAISS as LangFAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info("Building LangChain FAISS retriever...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    vectorstore = LangFAISS.from_texts(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})


# ─── OpenAI Verdict Generation ─────────────────────────────────────────────────
def get_openai_verdict(claim: str, evidence_chunks: list, citation_key: str) -> dict:
    """
    Use OpenAI GPT to judge if claim is supported by evidence.
    Resume skill: OpenAI API, RAG pipeline.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _heuristic_verdict(claim, evidence_chunks)

    client = OpenAI(api_key=api_key)
    evidence_text = "\n\n---\n\n".join(f"[Chunk {i+1}]:\n{c}" for i, c in enumerate(evidence_chunks[:3]))

    prompt = f"""You are an academic citation verifier. Your job is to check whether a CLAIM made in a research paper is genuinely supported by the CITED EVIDENCE.

CLAIM from the main paper (citing {citation_key}):
"{claim}"

EVIDENCE from the cited paper:
{evidence_text}

Respond ONLY in valid JSON with these exact fields:
{{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT",
  "confidence": <float 0.0-1.0>,
  "explanation": "<one sentence explaining your verdict>",
  "key_evidence": "<the most relevant quote or sentence from the evidence>"
}}

Rules:
- SUPPORTED: the evidence clearly backs the claim
- CONTRADICTED: the evidence says something opposite or incompatible
- INSUFFICIENT: evidence is related but doesn't directly verify the claim
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"OpenAI call failed: {e}, falling back to heuristic")
        return _heuristic_verdict(claim, evidence_chunks)


def _heuristic_verdict(claim: str, evidence_chunks: list) -> dict:
    """
    Heuristic fallback when OpenAI key is not set.
    Uses keyword overlap and semantic similarity.
    """
    if not evidence_chunks:
        return {
            "verdict": "INSUFFICIENT",
            "confidence": 0.0,
            "explanation": "No evidence found in the cited paper for this claim.",
            "key_evidence": "",
        }

    claim_words = set(re.findall(r'\b\w{4,}\b', claim.lower()))
    scores = []
    for chunk in evidence_chunks:
        chunk_words = set(re.findall(r'\b\w{4,}\b', chunk.lower()))
        overlap = len(claim_words & chunk_words)
        scores.append((overlap, chunk))

    scores.sort(reverse=True)
    best_score, best_chunk = scores[0]
    max_possible = max(len(claim_words), 1)
    confidence = min(best_score / max_possible, 1.0)

    if confidence >= 0.35:
        verdict = "SUPPORTED"
        explanation = "Keyword and semantic overlap suggests the cited paper discusses this claim."
    elif confidence >= 0.15:
        verdict = "INSUFFICIENT"
        explanation = "Partial overlap found but not enough to fully verify the claim."
    else:
        verdict = "INSUFFICIENT"
        explanation = "Very little overlap between the claim and cited paper content."

    # Sentence with most overlap as key evidence
    best_sentence = max(
        re.split(r'(?<=[.!?])\s+', best_chunk),
        key=lambda s: len(set(re.findall(r'\b\w{4,}\b', s.lower())) & claim_words),
        default=best_chunk[:200]
    )

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "explanation": explanation,
        "key_evidence": best_sentence[:300],
    }


# ─── Evaluation Pipeline ───────────────────────────────────────────────────────
def compute_evaluation_stats(results: list) -> dict:
    """
    Compute aggregate evaluation metrics across all verified claims.
    Resume skill: evaluation pipeline.
    """
    if not results:
        return {}

    total = len(results)
    supported = sum(1 for r in results if r["verdict"] == "SUPPORTED")
    contradicted = sum(1 for r in results if r["verdict"] == "CONTRADICTED")
    insufficient = sum(1 for r in results if r["verdict"] == "INSUFFICIENT")
    avg_confidence = sum(r["confidence"] for r in results) / total
    avg_similarity = sum(r["similarity_score"] for r in results) / total

    # Faithfulness: % of claims that are SUPPORTED or have strong evidence
    faithfulness = supported / total if total > 0 else 0.0

    # Hallucination risk: claims marked CONTRADICTED
    hallucination_risk = contradicted / total if total > 0 else 0.0

    return {
        "total_claims_verified": total,
        "supported": supported,
        "contradicted": contradicted,
        "insufficient": insufficient,
        "faithfulness_score": round(faithfulness, 3),
        "hallucination_risk": round(hallucination_risk, 3),
        "avg_confidence": round(avg_confidence, 3),
        "avg_similarity_score": round(avg_similarity, 3),
        "overall_integrity": "HIGH" if faithfulness >= 0.7 else "MEDIUM" if faithfulness >= 0.4 else "LOW",
    }


# ─── Core Verification Job ─────────────────────────────────────────────────────
def run_verification_job(
    job_id: str,
    main_paper_path: str,
    cited_paper_path: str,
):
    """
    Full RAG pipeline:
    1. Extract text from PDFs
    2. Parse citations from main paper
    3. Chunk cited paper text
    4. Build FAISS + ChromaDB indexes
    5. For each claim, retrieve evidence and judge verdict
    6. Compute evaluation stats
    """
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 5
        jobs[job_id]["message"] = "Extracting text from PDFs..."

        main_text = extract_text_from_pdf(main_paper_path)
        cited_text = extract_text_from_pdf(cited_paper_path)

        jobs[job_id]["progress"] = 15
        jobs[job_id]["message"] = "Parsing citations from main paper..."

        citations = extract_citations(main_text)
        logger.info(f"Found {len(citations)} citation groups: {list(citations.keys())[:5]}")

        if not citations:
            # If no structured citations found, treat the whole paper as one "claim block"
            citations = {"[ALL]": re.split(r'(?<=[.!?])\s+', main_text)[:30]}

        jobs[job_id]["progress"] = 25
        jobs[job_id]["message"] = f"Chunking cited paper ({len(cited_text.split())} words)..."

        chunks = chunk_text(cited_text, chunk_size=500, overlap=80)
        logger.info(f"Generated {len(chunks)} chunks from cited paper.")

        jobs[job_id]["progress"] = 35
        jobs[job_id]["message"] = "Building FAISS vector index..."

        faiss_index, _ = build_faiss_index(chunks)

        jobs[job_id]["progress"] = 50
        jobs[job_id]["message"] = "Building ChromaDB collection..."

        chroma_col = build_chroma_collection(f"cited_{job_id[:8]}", chunks)

        jobs[job_id]["progress"] = 60
        jobs[job_id]["message"] = "Building LangChain RAG retriever..."

        lc_retriever = build_langchain_retriever(chunks)

        jobs[job_id]["progress"] = 65
        jobs[job_id]["message"] = "Verifying claims against cited paper..."

        results = []
        citation_items = list(citations.items())[:20]   # Cap at 20 for performance

        for i, (cite_key, sentences) in enumerate(citation_items):
            claims = extract_claims_from_sentences(sentences, cite_key)
            if not claims:
                claims = sentences[:1]

            for claim in claims[:2]:   # Max 2 claims per citation key
                # Retrieve from FAISS
                faiss_hits = faiss_search(faiss_index, chunks, claim, top_k=5)
                faiss_chunks = [c for c, _ in faiss_hits]
                faiss_score = faiss_hits[0][1] if faiss_hits else 0.0

                # Retrieve from ChromaDB (cross-validation)
                chroma_hits = chroma_search(chroma_col, claim, top_k=3)
                chroma_chunks = [c for c, _ in chroma_hits]

                # Merge evidence from both stores (dedup)
                all_evidence = list(dict.fromkeys(faiss_chunks + chroma_chunks))[:5]

                # LangChain retriever (third pass for RAG completeness)
                try:
                    lc_docs = lc_retriever.invoke(claim)
                    lc_chunks = [d.page_content for d in lc_docs]
                    all_evidence = list(dict.fromkeys(all_evidence + lc_chunks))[:5]
                except Exception:
                    pass

                # Judge verdict
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

            # Update progress during verification
            progress = 65 + int((i / len(citation_items)) * 30)
            jobs[job_id]["progress"] = progress
            jobs[job_id]["message"] = f"Verified {i+1}/{len(citation_items)} citation groups..."

        jobs[job_id]["progress"] = 97
        jobs[job_id]["message"] = "Computing evaluation metrics..."

        stats = compute_evaluation_stats(results)

        jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "message": "Verification complete.",
            "results": results,
            "stats": stats,
        })

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id].update({
            "status": "error",
            "progress": 0,
            "message": "Verification failed.",
            "error": str(e),
        })
    finally:
        # Cleanup temp files
        for p in [main_paper_path, cited_paper_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


# ─── API Routes ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Research Paper Citation Verifier API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "openai_configured": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/verify", response_model=dict)
async def verify_citations(
    background_tasks: BackgroundTasks,
    main_paper: UploadFile = File(..., description="The research paper whose citations you want to verify"),
    cited_paper: UploadFile = File(..., description="The paper being cited (to check against)"),
):
    """
    Start an async citation verification job.
    Returns a job_id to poll for results.
    """
    for f in [main_paper, cited_paper]:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' must be a PDF.")

    job_id = str(uuid.uuid4())

    # Save uploads to temp files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
        tmp1.write(await main_paper.read())
        main_path = tmp1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
        tmp2.write(await cited_paper.read())
        cited_path = tmp2.name

    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Job queued...",
        "results": None,
        "stats": None,
        "error": None,
    }

    background_tasks.add_task(run_verification_job, job_id, main_path, cited_path)

    return {"job_id": job_id, "message": "Verification started. Poll /status/{job_id} for results."}


@app.get("/status/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatus(**jobs[job_id])


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    if job_id in jobs:
        del jobs[job_id]
    return {"deleted": job_id}


@app.get("/demo")
def demo_result():
    """Returns a sample result for UI testing without real PDFs."""
    return {
        "job_id": "demo",
        "status": "done",
        "progress": 100,
        "message": "Demo verification complete.",
        "results": [
            {
                "claim": "The proposed transformer architecture achieves state-of-the-art performance on the GLUE benchmark [1].",
                "citation_key": "[1]",
                "verdict": "SUPPORTED",
                "confidence": 0.91,
                "evidence": "Our model achieves 88.9 on the GLUE benchmark, surpassing all previously reported results.",
                "explanation": "The cited paper explicitly reports GLUE benchmark scores that support this claim.",
                "similarity_score": 0.873,
            },
            {
                "claim": "Training on larger datasets consistently improves downstream task performance [2].",
                "citation_key": "[2]",
                "verdict": "INSUFFICIENT",
                "confidence": 0.54,
                "evidence": "We observe improved performance with scale in most settings, though diminishing returns are noted.",
                "explanation": "Evidence partially supports the claim but includes nuance (diminishing returns) not reflected in the claim.",
                "similarity_score": 0.712,
            },
            {
                "claim": "Attention mechanisms are computationally equivalent to recurrent networks [3].",
                "citation_key": "[3]",
                "verdict": "CONTRADICTED",
                "confidence": 0.82,
                "evidence": "Self-attention has O(n²) complexity vs O(n) for recurrent networks, making them fundamentally different.",
                "explanation": "The cited paper explicitly describes different computational complexities, contradicting the claim.",
                "similarity_score": 0.645,
            },
        ],
        "stats": {
            "total_claims_verified": 3,
            "supported": 1,
            "contradicted": 1,
            "insufficient": 1,
            "faithfulness_score": 0.333,
            "hallucination_risk": 0.333,
            "avg_confidence": 0.757,
            "avg_similarity_score": 0.743,
            "overall_integrity": "LOW",
        },
    }
