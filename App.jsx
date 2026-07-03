import React, { useState, useCallback, useEffect, useRef } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import {
  UploadCloud, FileText, CheckCircle2, XCircle, AlertCircle,
  Loader2, ChevronDown, ChevronUp, BarChart2, RefreshCw,
  Beaker, ExternalLink, Shield, Zap, Database, Search
} from "lucide-react";
import {
  RadialBarChart, RadialBar, PieChart, Pie, Cell,
  Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid
} from "recharts";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000";

// ─── Color constants ───────────────────────────────────────────────────────────
const COLORS = {
  SUPPORTED: { bg: "#e8f5e9", text: "#1b5e20", border: "#4caf50", icon: "#2e7d32" },
  CONTRADICTED: { bg: "#fce4ec", text: "#880e4f", border: "#e91e63", icon: "#c62828" },
  INSUFFICIENT: { bg: "#fff8e1", text: "#e65100", border: "#ff9800", icon: "#f57c00" },
};

const PIE_COLORS = ["#4caf50", "#e91e63", "#ff9800"];

// ─── File Drop Zone ────────────────────────────────────────────────────────────
function DropZone({ label, file, onFile, accent }) {
  const onDrop = useCallback(files => {
    if (files[0]) onFile(files[0]);
  }, [onFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
  });

  return (
    <div
      {...getRootProps()}
      style={{
        border: `2px dashed ${isDragActive ? accent : "#cbd5e1"}`,
        borderRadius: 12,
        padding: "28px 20px",
        textAlign: "center",
        cursor: "pointer",
        background: isDragActive ? `${accent}10` : "#f8fafc",
        transition: "all 0.2s",
        minHeight: 140,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 10,
      }}
    >
      <input {...getInputProps()} />
      {file ? (
        <>
          <FileText size={32} color={accent} />
          <div style={{ fontWeight: 600, color: "#1e293b", fontSize: 14 }}>{file.name}</div>
          <div style={{ fontSize: 12, color: "#64748b" }}>
            {(file.size / 1024).toFixed(1)} KB · Click to replace
          </div>
        </>
      ) : (
        <>
          <UploadCloud size={32} color={isDragActive ? accent : "#94a3b8"} />
          <div style={{ fontWeight: 600, color: "#475569", fontSize: 14 }}>{label}</div>
          <div style={{ fontSize: 12, color: "#94a3b8" }}>Drop PDF here or click to browse</div>
        </>
      )}
    </div>
  );
}

// ─── Verdict Badge ─────────────────────────────────────────────────────────────
function VerdictBadge({ verdict }) {
  const c = COLORS[verdict] || COLORS.INSUFFICIENT;
  const icons = {
    SUPPORTED: <CheckCircle2 size={14} />,
    CONTRADICTED: <XCircle size={14} />,
    INSUFFICIENT: <AlertCircle size={14} />,
  };
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      background: c.bg, color: c.text, border: `1px solid ${c.border}`,
      borderRadius: 20, padding: "3px 10px", fontSize: 12, fontWeight: 600,
    }}>
      {icons[verdict]} {verdict}
    </span>
  );
}

// ─── Confidence Bar ────────────────────────────────────────────────────────────
function ConfidenceBar({ value, verdict }) {
  const c = COLORS[verdict] || COLORS.INSUFFICIENT;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        flex: 1, height: 6, background: "#e2e8f0", borderRadius: 3, overflow: "hidden"
      }}>
        <div style={{
          width: `${Math.round(value * 100)}%`, height: "100%",
          background: c.border, borderRadius: 3, transition: "width 0.6s ease",
        }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 600, color: "#64748b", minWidth: 36 }}>
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}

// ─── Result Card ───────────────────────────────────────────────────────────────
function ResultCard({ result, index }) {
  const [expanded, setExpanded] = useState(false);
  const c = COLORS[result.verdict] || COLORS.INSUFFICIENT;

  return (
    <div style={{
      border: `1px solid ${c.border}40`,
      borderLeft: `4px solid ${c.border}`,
      borderRadius: 10, background: "#fff",
      marginBottom: 12, overflow: "hidden",
      boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
    }}>
      <div
        style={{ padding: "14px 16px", cursor: "pointer" }}
        onClick={() => setExpanded(e => !e)}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <span style={{
                fontSize: 11, fontWeight: 700, color: "#64748b",
                background: "#f1f5f9", padding: "2px 8px", borderRadius: 4,
              }}>
                {result.citation_key}
              </span>
              <VerdictBadge verdict={result.verdict} />
            </div>
            <p style={{ margin: 0, fontSize: 14, color: "#1e293b", lineHeight: 1.5 }}>
              {result.claim.length > 180 ? result.claim.slice(0, 180) + "…" : result.claim}
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 3 }}>Similarity</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#334155" }}>
                {Math.round(result.similarity_score * 100)}%
              </div>
            </div>
            {expanded ? <ChevronUp size={18} color="#94a3b8" /> : <ChevronDown size={18} color="#94a3b8" />}
          </div>
        </div>
        <div style={{ marginTop: 8 }}>
          <ConfidenceBar value={result.confidence} verdict={result.verdict} />
        </div>
      </div>

      {expanded && (
        <div style={{ padding: "0 16px 16px", borderTop: `1px solid ${c.border}20` }}>
          <div style={{ paddingTop: 14, display: "flex", flexDirection: "column", gap: 12 }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
                AI Explanation
              </div>
              <p style={{ margin: 0, fontSize: 13, color: "#475569", lineHeight: 1.6 }}>
                {result.explanation}
              </p>
            </div>
            {result.evidence && (
              <div style={{
                background: "#f8fafc", borderRadius: 8, padding: "10px 14px",
                borderLeft: "3px solid #94a3b8",
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
                  Key Evidence from Cited Paper
                </div>
                <p style={{ margin: 0, fontSize: 13, color: "#374151", lineHeight: 1.6, fontStyle: "italic" }}>
                  "{result.evidence}"
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Stats Panel ───────────────────────────────────────────────────────────────
function StatsPanel({ stats }) {
  if (!stats) return null;

  const pieData = [
    { name: "Supported", value: stats.supported },
    { name: "Contradicted", value: stats.contradicted },
    { name: "Insufficient", value: stats.insufficient },
  ].filter(d => d.value > 0);

  const barData = [
    { name: "Faithfulness", value: Math.round(stats.faithfulness_score * 100) },
    { name: "Confidence", value: Math.round(stats.avg_confidence * 100) },
    { name: "Similarity", value: Math.round(stats.avg_similarity_score * 100) },
    { name: "Halluc. Risk", value: Math.round(stats.hallucination_risk * 100) },
  ];

  const integrityColor = {
    HIGH: "#4caf50", MEDIUM: "#ff9800", LOW: "#e91e63"
  }[stats.overall_integrity] || "#94a3b8";

  return (
    <div style={{
      background: "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)",
      border: "1px solid #e2e8f0", borderRadius: 14, padding: 24, marginBottom: 24,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 20 }}>
        <BarChart2 size={18} color="#6366f1" />
        <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: "#1e293b" }}>
          Verification Report
        </h3>
        <span style={{
          marginLeft: "auto", padding: "4px 14px", borderRadius: 20,
          background: integrityColor + "20", color: integrityColor,
          fontSize: 12, fontWeight: 700, border: `1px solid ${integrityColor}40`,
        }}>
          {stats.overall_integrity} INTEGRITY
        </span>
      </div>

      {/* Metric cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 24 }}>
        {[
          { label: "Claims Verified", value: stats.total_claims_verified, icon: <Search size={16} /> },
          { label: "Supported", value: stats.supported, icon: <CheckCircle2 size={16} color="#4caf50" /> },
          { label: "Contradicted", value: stats.contradicted, icon: <XCircle size={16} color="#e91e63" /> },
          { label: "Insufficient", value: stats.insufficient, icon: <AlertCircle size={16} color="#ff9800" /> },
        ].map(m => (
          <div key={m.label} style={{
            background: "#fff", borderRadius: 10, padding: "14px 16px",
            border: "1px solid #e2e8f0", textAlign: "center",
          }}>
            <div style={{ display: "flex", justifyContent: "center", marginBottom: 6, color: "#94a3b8" }}>
              {m.icon}
            </div>
            <div style={{ fontSize: 24, fontWeight: 700, color: "#1e293b" }}>{m.value}</div>
            <div style={{ fontSize: 11, color: "#94a3b8", fontWeight: 500 }}>{m.label}</div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#64748b", marginBottom: 10 }}>
            Verdict Distribution
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" outerRadius={60} dataKey="value" label={({ name, percent }) => `${name} ${Math.round(percent * 100)}%`} labelLine={false} fontSize={11}>
                {pieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#64748b", marginBottom: 10 }}>
            Quality Metrics (%)
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={barData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => `${v}%`} />
              <Bar dataKey="value" radius={4} fill="#6366f1" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ─── Pipeline Badges ───────────────────────────────────────────────────────────
function TechBadge({ icon, label }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      background: "#f1f5f9", border: "1px solid #e2e8f0",
      borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 600, color: "#475569",
    }}>
      {icon} {label}
    </span>
  );
}

// ─── Progress Bar ──────────────────────────────────────────────────────────────
function ProgressBar({ progress, message }) {
  return (
    <div style={{ padding: "24px 0", textAlign: "center" }}>
      <Loader2 size={32} color="#6366f1" style={{ animation: "spin 1s linear infinite", marginBottom: 12 }} />
      <div style={{ fontSize: 14, fontWeight: 600, color: "#1e293b", marginBottom: 8 }}>
        {message}
      </div>
      <div style={{
        background: "#e2e8f0", borderRadius: 10, height: 8,
        overflow: "hidden", maxWidth: 400, margin: "0 auto",
      }}>
        <div style={{
          width: `${progress}%`, height: "100%",
          background: "linear-gradient(90deg, #6366f1, #8b5cf6)",
          borderRadius: 10, transition: "width 0.4s ease",
        }} />
      </div>
      <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 6 }}>{progress}%</div>
    </div>
  );
}

// ─── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [mainPaper, setMainPaper] = useState(null);
  const [citedPaper, setCitedPaper] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [error, setError] = useState("");
  const [filterVerdict, setFilterVerdict] = useState("ALL");
  const [loadingDemo, setLoadingDemo] = useState(false);
  const pollRef = useRef(null);

  // Poll job status
  useEffect(() => {
    if (!jobId) return;

    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/status/${jobId}`);
        setJobStatus(res.data);
        if (["done", "error"].includes(res.data.status)) {
          clearInterval(pollRef.current);
        }
      } catch (e) {
        clearInterval(pollRef.current);
        setError("Lost connection to backend.");
      }
    }, 1500);

    return () => clearInterval(pollRef.current);
  }, [jobId]);

  const handleVerify = async () => {
    if (!mainPaper || !citedPaper) {
      setError("Please upload both PDFs before verifying.");
      return;
    }
    setError("");
    setJobStatus(null);
    setJobId(null);

    const form = new FormData();
    form.append("main_paper", mainPaper);
    form.append("cited_paper", citedPaper);

    try {
      const res = await axios.post(`${API}/verify`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setJobId(res.data.job_id);
      setJobStatus({ status: "pending", progress: 0, message: "Job queued..." });
    } catch (e) {
      setError(e.response?.data?.detail || "Failed to start verification. Is the backend running?");
    }
  };

  const handleDemo = async () => {
    setLoadingDemo(true);
    setError("");
    setJobId(null);
    try {
      const res = await axios.get(`${API}/demo`);
      setJobStatus({ ...res.data, status: "done" });
    } catch (e) {
      setError("Could not reach backend for demo.");
    } finally {
      setLoadingDemo(false);
    }
  };

  const handleReset = () => {
    setMainPaper(null);
    setCitedPaper(null);
    setJobId(null);
    setJobStatus(null);
    setError("");
    setFilterVerdict("ALL");
    if (pollRef.current) clearInterval(pollRef.current);
  };

  const results = jobStatus?.results || [];
  const filteredResults = filterVerdict === "ALL"
    ? results
    : results.filter(r => r.verdict === filterVerdict);

  const isProcessing = jobStatus && !["done", "error"].includes(jobStatus.status);

  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", fontFamily: "'Inter', system-ui, sans-serif" }}>
      {/* Header */}
      <div style={{
        background: "#fff", borderBottom: "1px solid #e2e8f0",
        padding: "16px 0", position: "sticky", top: 0, zIndex: 50,
      }}>
        <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 24px", display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 9,
            background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Shield size={18} color="#fff" />
          </div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>
              CitationVerifier
            </div>
            <div style={{ fontSize: 11, color: "#94a3b8" }}>
              RAG-powered academic citation checker
            </div>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 6, flexWrap: "wrap" }}>
            {[
              [<Database size={11} />, "LangChain"],
              [<Database size={11} />, "ChromaDB"],
              [<Zap size={11} />, "FAISS"],
              [<Beaker size={11} />, "SentenceTransformers"],
            ].map(([icon, label]) => (
              <TechBadge key={label} icon={icon} label={label} />
            ))}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "32px 24px" }}>
        {/* Upload section */}
        {!isProcessing && jobStatus?.status !== "done" && (
          <div style={{
            background: "#fff", borderRadius: 16, border: "1px solid #e2e8f0",
            padding: 28, marginBottom: 24, boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
          }}>
            <h2 style={{ margin: "0 0 6px", fontSize: 18, fontWeight: 700, color: "#0f172a" }}>
              Upload Research Papers
            </h2>
            <p style={{ margin: "0 0 24px", color: "#64748b", fontSize: 14 }}>
              Upload the <strong>main paper</strong> (whose claims you want to verify) and the <strong>cited paper</strong> it references.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#6366f1", marginBottom: 8 }}>
                  📄 MAIN PAPER (with citations)
                </div>
                <DropZone
                  label="Main Research Paper"
                  file={mainPaper}
                  onFile={setMainPaper}
                  accent="#6366f1"
                />
              </div>
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#0ea5e9", marginBottom: 8 }}>
                  📑 CITED PAPER (to verify against)
                </div>
                <DropZone
                  label="Cited Reference Paper"
                  file={citedPaper}
                  onFile={setCitedPaper}
                  accent="#0ea5e9"
                />
              </div>
            </div>

            {error && (
              <div style={{
                background: "#fce4ec", border: "1px solid #f48fb1",
                borderRadius: 8, padding: "10px 14px", marginBottom: 16,
                fontSize: 13, color: "#880e4f",
              }}>
                ⚠ {error}
              </div>
            )}

            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <button
                onClick={handleVerify}
                disabled={!mainPaper || !citedPaper}
                style={{
                  background: (!mainPaper || !citedPaper) ? "#e2e8f0" : "linear-gradient(135deg, #6366f1, #8b5cf6)",
                  color: (!mainPaper || !citedPaper) ? "#94a3b8" : "#fff",
                  border: "none", borderRadius: 10, padding: "11px 28px",
                  fontSize: 14, fontWeight: 600, cursor: (!mainPaper || !citedPaper) ? "not-allowed" : "pointer",
                  display: "flex", alignItems: "center", gap: 8,
                }}
              >
                <Search size={16} /> Verify Citations
              </button>

              <button
                onClick={handleDemo}
                disabled={loadingDemo}
                style={{
                  background: "#fff", color: "#6366f1",
                  border: "1px solid #c7d2fe", borderRadius: 10, padding: "11px 20px",
                  fontSize: 14, fontWeight: 600, cursor: "pointer",
                  display: "flex", alignItems: "center", gap: 8,
                }}
              >
                {loadingDemo ? <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} /> : <Beaker size={16} />}
                Try Demo
              </button>
            </div>
          </div>
        )}

        {/* Processing */}
        {isProcessing && (
          <div style={{
            background: "#fff", borderRadius: 16, border: "1px solid #e2e8f0",
            padding: 28, marginBottom: 24,
          }}>
            <ProgressBar progress={jobStatus.progress} message={jobStatus.message} />
            <div style={{ textAlign: "center", marginTop: 12, display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 8 }}>
              {["Text Extraction", "Citation Parsing", "Text Chunking", "FAISS Index", "ChromaDB", "LangChain RAG", "Verdict Generation", "Eval Metrics"].map((step, i) => (
                <span key={step} style={{
                  fontSize: 11, padding: "3px 10px", borderRadius: 20,
                  background: jobStatus.progress > (i * 12) ? "#ede9fe" : "#f1f5f9",
                  color: jobStatus.progress > (i * 12) ? "#6366f1" : "#94a3b8",
                  fontWeight: 600,
                }}>
                  {jobStatus.progress > (i * 12) ? "✓ " : ""}{step}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {jobStatus?.status === "error" && (
          <div style={{
            background: "#fce4ec", border: "1px solid #f48fb1",
            borderRadius: 14, padding: 20, marginBottom: 24, textAlign: "center",
          }}>
            <XCircle size={32} color="#e91e63" style={{ marginBottom: 10 }} />
            <div style={{ fontSize: 15, fontWeight: 600, color: "#880e4f", marginBottom: 6 }}>Verification Failed</div>
            <div style={{ fontSize: 13, color: "#c2185b" }}>{jobStatus.error}</div>
            <button onClick={handleReset} style={{
              marginTop: 14, background: "#fff", color: "#880e4f",
              border: "1px solid #f48fb1", borderRadius: 8, padding: "8px 20px",
              fontSize: 13, fontWeight: 600, cursor: "pointer",
            }}>
              Try Again
            </button>
          </div>
        )}

        {/* Results */}
        {jobStatus?.status === "done" && results.length > 0 && (
          <>
            <StatsPanel stats={jobStatus.stats} />

            {/* Filter + reset */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: "#475569" }}>Filter:</span>
              {["ALL", "SUPPORTED", "CONTRADICTED", "INSUFFICIENT"].map(v => (
                <button
                  key={v}
                  onClick={() => setFilterVerdict(v)}
                  style={{
                    padding: "5px 14px", borderRadius: 20, fontSize: 12, fontWeight: 600,
                    border: "1px solid",
                    borderColor: filterVerdict === v ? "#6366f1" : "#e2e8f0",
                    background: filterVerdict === v ? "#ede9fe" : "#fff",
                    color: filterVerdict === v ? "#6366f1" : "#64748b",
                    cursor: "pointer",
                  }}
                >
                  {v} {v !== "ALL" && `(${results.filter(r => r.verdict === v).length})`}
                </button>
              ))}
              <button onClick={handleReset} style={{
                marginLeft: "auto", display: "flex", alignItems: "center", gap: 6,
                padding: "5px 14px", borderRadius: 20, fontSize: 12, fontWeight: 600,
                border: "1px solid #e2e8f0", background: "#fff", color: "#64748b", cursor: "pointer",
              }}>
                <RefreshCw size={13} /> New Verification
              </button>
            </div>

            {/* Result cards */}
            <div>
              {filteredResults.map((result, i) => (
                <ResultCard key={i} result={result} index={i} />
              ))}
              {filteredResults.length === 0 && (
                <div style={{ textAlign: "center", padding: 40, color: "#94a3b8" }}>
                  No claims with verdict "{filterVerdict}"
                </div>
              )}
            </div>
          </>
        )}

        {/* Pipeline info */}
        {!jobStatus && !isProcessing && (
          <div style={{
            background: "#fff", borderRadius: 14, border: "1px solid #e2e8f0",
            padding: 24, marginTop: 24,
          }}>
            <h3 style={{ margin: "0 0 16px", fontSize: 14, fontWeight: 700, color: "#1e293b" }}>
              🔬 Under the Hood — RAG Pipeline
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
              {[
                { icon: "📄", title: "PDF Extraction", desc: "pdfplumber + PyMuPDF for robust text extraction from research PDFs" },
                { icon: "✂️", title: "Text Chunking", desc: "LangChain RecursiveCharacterTextSplitter with 500-token chunks and 80-token overlap" },
                { icon: "🧠", title: "Embedding Generation", desc: "Sentence Transformers (all-MiniLM-L6-v2) for semantic embeddings" },
                { icon: "⚡", title: "FAISS Index", desc: "Fast Approximate Nearest Neighbor search for real-time semantic retrieval" },
                { icon: "🗄️", title: "ChromaDB", desc: "Persistent vector store for cross-validated retrieval with cosine similarity" },
                { icon: "🔗", title: "LangChain RAG", desc: "Full RAG pipeline with LangChain retriever for comprehensive evidence gathering" },
                { icon: "🤖", title: "Verdict Generation", desc: "GPT-4o-mini judges if claims are SUPPORTED / CONTRADICTED / INSUFFICIENT" },
                { icon: "📊", title: "Evaluation Pipeline", desc: "Faithfulness score, hallucination risk, and confidence metrics per claim" },
              ].map(step => (
                <div key={step.title} style={{
                  background: "#f8fafc", borderRadius: 10, padding: "12px 14px",
                  border: "1px solid #e2e8f0",
                }}>
                  <div style={{ fontSize: 18, marginBottom: 6 }}>{step.icon}</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "#1e293b", marginBottom: 4 }}>{step.title}</div>
                  <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.5 }}>{step.desc}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        * { box-sizing: border-box; }
        body { margin: 0; }
      `}</style>
    </div>
  );
}
