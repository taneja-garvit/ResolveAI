import { useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const samplePrompts = [
  "What is the refund policy in the uploaded handbook?",
  "Create a support ticket for a delayed shipment.",
  "Check the status of order 1024.",
];

export default function App() {
  const [question, setQuestion] = useState(samplePrompts[0]);
  const [answer, setAnswer] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [error, setError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);

  const healthUrl = useMemo(() => `${API_BASE_URL}/health`, []);

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setIsUploading(true);
    setUploadStatus("");
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/upload-doc`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Document upload failed.");
      }

      setUploadStatus(`Indexed ${data.filename || file.name} successfully.`);
    } catch (err) {
      setError(err.message || "Document upload failed.");
    } finally {
      setIsUploading(false);
      event.target.value = "";
    }
  }

  async function handleAsk(event) {
    event.preventDefault();
    setIsAsking(true);
    setAnswer("");
    setError("");
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000);

    try {
      const response = await fetch(
        `${API_BASE_URL}/ask?query=${encodeURIComponent(question)}`,
        {
          method: "POST",
          signal: controller.signal,
        }
      );

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Question failed.");
      }

      setAnswer(data.answer || "No answer returned.");
    } catch (err) {
      if (err.name === "AbortError") {
        setError("Request timed out. Please try a shorter query.");
        return;
      }
      setError(err.message || "Question failed.");
    } finally {
      clearTimeout(timeoutId);
      setIsAsking(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="hero">
        <p className="eyebrow">AI Customer Support Copilot</p>
        <h1>ResolveAI</h1>
        <p className="hero-copy">
          Upload company PDFs, ask support questions, and test the agent workflow
          from a simple React dashboard.
        </p>
      </div>

      <div className="grid">
        <section className="card">
          <h2>1. Backend Check</h2>
          <p className="muted">
            Make sure the FastAPI backend is running before testing uploads and
            queries.
          </p>
          <a href={healthUrl} target="_blank" rel="noreferrer" className="link">
            Open health endpoint
          </a>
        </section>

        <section className="card">
          <h2>2. Upload PDF</h2>
          <p className="muted">
            Upload a support handbook, FAQ, or company policy document to build the
            FAISS vector store.
          </p>
          <label className="upload-box">
            <span>{isUploading ? "Uploading..." : "Choose a PDF file"}</span>
            <input type="file" accept="application/pdf" onChange={handleUpload} />
          </label>
          {uploadStatus ? <p className="success">{uploadStatus}</p> : null}
        </section>

        <section className="card card-wide">
          <h2>3. Ask the Copilot</h2>
          <form onSubmit={handleAsk} className="question-form">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={5}
              placeholder="Ask about refunds, order policies, or uploaded company docs..."
            />
            <div className="prompt-row">
              {samplePrompts.map((prompt) => (
                <button
                  type="button"
                  key={prompt}
                  className="chip"
                  onClick={() => setQuestion(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
            <button type="submit" className="primary-button" disabled={isAsking}>
              {isAsking ? "Thinking..." : "Ask ResolveAI"}
            </button>
          </form>

          {error ? <p className="error">{error}</p> : null}

          <div className="answer-panel">
            <h3>Response</h3>
            <pre>{answer || "Your answer will appear here."}</pre>
          </div>
        </section>
      </div>
    </div>
  );
}
