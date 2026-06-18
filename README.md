# 📚 Research Paper Citation Verifier

> RAG pipeline that checks whether a paper's claims are genuinely supported by the papers it cites.

**Resume-worthy stack:** LangChain · ChromaDB · FAISS · RAG Pipeline · OpenAI API · FastAPI · Sentence Transformers · Vector Database · Evaluation Pipeline · Text Chunking · Embedding Generation

---

## 🏗️ Architecture

```
User uploads 2 PDFs
        │
        ▼
┌─────────────────────────────────────────────┐
│              FastAPI Backend                │
│                                             │
│  1. PDF Extraction (pdfplumber + PyMuPDF)   │
│  2. Citation Parser (regex → claim extractor│
│  3. Text Chunking (LangChain splitter)      │
│  4. Embedding Generation (SentenceTransformers all-MiniLM-L6-v2)
│  5. FAISS Index (fast ANN search)           │
│  6. ChromaDB Collection (persistent store)  │
│  7. LangChain RAG Retriever                 │
│  8. OpenAI GPT-4o-mini verdict generation   │
│     (heuristic fallback if no key)          │
│  9. Evaluation metrics (faithfulness, etc.) │
└─────────────────────────────────────────────┘
        │
        ▼
   JSON Results → React UI
```

---

## 🚀 Quick Start

### Option 1: Docker Compose (recommended)

```bash
git clone <this-repo>
cd citation-verifier

# Optional: add your OpenAI key for LLM verdicts
echo "OPENAI_API_KEY=sk-..." > .env

docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

### Option 2: Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env to add your OPENAI_API_KEY (optional)

uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
REACT_APP_API_URL=http://localhost:8000 npm start
```

---

## 📋 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/verify` | Upload 2 PDFs, start verification job |
| `GET` | `/status/{job_id}` | Poll job progress and results |
| `GET` | `/demo` | Get sample results (no PDFs needed) |
| `GET` | `/health` | Check if backend is running |
| `GET` | `/docs` | Auto-generated Swagger UI |

**Upload example:**
```bash
curl -X POST http://localhost:8000/verify \
  -F "main_paper=@my_paper.pdf" \
  -F "cited_paper=@reference.pdf"
```

---

## 🎯 How It Works

### Step 1 — PDF Extraction
Uses `pdfplumber` (primary) with `pymupdf` fallback to extract full text from both PDFs.

### Step 2 — Citation Parsing
Regex-based parser detects:
- Numeric citations: `[1]`, `[2,3]`, `[1-5]`
- Author-year citations: `(Smith, 2020)`, `(Jones et al., 2019)`

For each citation, surrounding sentences are extracted as potential claims.

### Step 3 — Text Chunking
`LangChain RecursiveCharacterTextSplitter` splits the cited paper into 500-token chunks with 80-token overlap, preserving sentence boundaries.

### Step 4 — Embedding Generation
`sentence-transformers/all-MiniLM-L6-v2` generates 384-dimensional normalized embeddings for all chunks.

### Step 5 — Dual Vector Store
- **FAISS** (`IndexFlatIP`): Fast inner-product search over all chunk embeddings
- **ChromaDB**: Persistent collection with cosine similarity metadata

Both stores are queried independently and results are merged/deduplicated for robustness.

### Step 6 — LangChain RAG Retriever
A third retrieval pass using `LangChain FAISS` vectorstore wrapped in a retriever, giving the pipeline full LangChain compatibility for future extensions (e.g., ReAct agents, chain-of-thought).

### Step 7 — Verdict Generation
Either:
- **OpenAI GPT-4o-mini**: Structured JSON verdict with confidence and key evidence
- **Heuristic fallback**: Keyword overlap + semantic similarity when no API key

Each claim gets one of: `SUPPORTED` | `CONTRADICTED` | `INSUFFICIENT`

### Step 8 — Evaluation Metrics
```python
faithfulness_score    = supported / total_claims
hallucination_risk    = contradicted / total_claims
avg_confidence        = mean(confidence scores)
avg_similarity_score  = mean(FAISS similarity scores)
overall_integrity     = HIGH / MEDIUM / LOW
```

---

## 📁 Project Structure

```
citation-verifier/
├── backend/
│   ├── main.py              # FastAPI app + full RAG pipeline
│   ├── requirements.txt     # All dependencies
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Complete React UI
│   │   └── index.js
│   ├── public/index.html
│   └── package.json
├── docker-compose.yml
└── README.md
```

---

## 🔑 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(empty)* | GPT-4o-mini for LLM verdicts (optional) |
| `CHUNK_SIZE` | `500` | Token size per text chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between consecutive chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model |

---

## ☁️ Cloud Deployment

### Deploy Backend to Railway
```bash
# In backend/
railway init
railway up
```

### Deploy Frontend to Vercel
```bash
cd frontend
vercel --env REACT_APP_API_URL=https://your-backend.railway.app
```

### Deploy Frontend to Netlify
```bash
cd frontend
npm run build
netlify deploy --prod --dir=build
```

---

## 🧪 Tech Stack (Resume Keywords)

| Library | Role |
|---------|------|
| **LangChain** | RAG pipeline, document splitting, retriever abstraction |
| **ChromaDB** | Persistent vector database with cosine similarity |
| **FAISS** | High-speed approximate nearest neighbor vector search |
| **Sentence Transformers** | Local embedding generation (no API cost) |
| **OpenAI API** | LLM-based claim verdict generation |
| **FastAPI** | Async REST API with background task processing |
| **RAG Pipeline** | Full retrieval-augmented generation workflow |
| **Vector Database** | Dual-store (FAISS + ChromaDB) for robust retrieval |
| **Evaluation Pipeline** | Faithfulness, hallucination risk, confidence metrics |
| **Text Chunking** | Recursive splitting with configurable overlap |
| **Embedding Generation** | Normalized dense vector representations |

---

## 💡 Ideas to Extend

- Add **RAGAS** evaluation library for standardized RAG metrics
- Support **multiple cited papers** (many-to-one verification)
- Add **DOI/arXiv fetching** to auto-download cited PDFs
- Export results as PDF report
- Add **user authentication** and job history database
