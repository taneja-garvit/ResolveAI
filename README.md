# ResolveAI

ResolveAI is an AI customer support copilot that combines retrieval-augmented generation, an agent-based decision workflow, and a lightweight confidence model to answer support queries over company documents.

Users can upload PDF knowledge-base documents, ask questions grounded in those documents, and trigger support-style actions such as refund processing, order lookup, or ticket creation through an agent with tool calling. A logistic regression model estimates confidence from retrieval quality and query length so uncertain responses can be escalated.

## Why This Project Stands Out

- End-to-end AI workflow instead of a single chatbot endpoint
- RAG pipeline over uploaded company documents using FAISS
- Agent-driven support actions with tool selection
- Confidence-based escalation layer for safer responses
- React frontend for quick local demos and recruiter walkthroughs

## Current Scope

This is a portfolio-grade prototype. The refund, order, and ticket tools are currently demo actions that return mock responses instead of hitting real production systems. That is fine for a resume project as long as the integrations are described honestly.

## Architecture

1. A user uploads a PDF through the frontend or `POST /upload-doc`.
2. FastAPI stores the document, chunks it, creates embeddings, and saves vectors in FAISS.
3. A user asks a support question through the React UI or `POST /ask`.
4. The agent decides whether to answer from retrieved documents or call a support tool.
5. Retrieval scores are converted into a normalized similarity signal.
6. A logistic regression model predicts response confidence.
7. Low-confidence cases are escalated instead of answered directly.

## Tech Stack

### Backend

- FastAPI
- LangChain
- Groq (OpenAI-compatible chat API)
- Sentence-Transformers embeddings
- FAISS
- scikit-learn
- pytest

### Frontend

- React
- Vite
- Fetch API

## Folder Structure

```text
app/         FastAPI backend, RAG pipeline, agent tools, ML scoring
frontend/    React demo interface for uploads and asking questions
tests/       Backend tests
data/        Uploaded PDFs and FAISS vector store
```

## Run Locally

### 1. Backend setup

Create and activate a virtual environment, then install Python dependencies:

```bash
pip install -r requirements.txt
```

Create the backend env file:

```bash
copy .env.example .env
```

Then set your `GROQ_API_KEY` in `.env` or your shell environment.

Train the confidence model:

```bash
python train_model.py
```

Start the backend:

```bash
uvicorn app.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`.

Environment variables used by backend:

- `GROQ_API_KEY`
- `GROQ_MODEL` (default: `llama-3.1-8b-instant`)
- `GROQ_BASE_URL` (default: `https://api.groq.com/openai/v1`)
- `EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)

### 2. Frontend setup

Open a second terminal:

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Frontend runs at `http://127.0.0.1:5173`.

If needed, set `VITE_API_URL` in `frontend/.env` to point at a different backend URL.

## Quick Test Flow

1. Start the backend.
2. Start the frontend.
3. Open `http://127.0.0.1:5173`.
4. Upload a PDF handbook or FAQ.
5. Ask a policy question such as "What is the refund policy?"
6. Try a tool-style query such as "Create a support ticket for a delayed shipment."

## API Endpoints

- `GET /health` - health check
- `POST /upload-doc` - upload a PDF document for indexing
- `POST /ask` - ask a question through the agent
- `GET /stream-ask` - stream an answer

## Tests

Run backend tests:

```bash
pytest
```

Run frontend build validation:

```bash
cd frontend
npm run build
```

## Resume Bullets

- Built an AI customer support copilot using `FastAPI`, `LangChain`, and `FAISS` to answer questions over uploaded company documents with retrieval-augmented generation.
- Added an agent-based workflow with tool calling for support operations and a `scikit-learn` logistic regression model to estimate confidence and trigger escalation for uncertain responses.

## Interview Explanation

"I wanted to build something more realistic than a basic chatbot, so I designed an AI customer support copilot with three layers. First, I used a RAG pipeline so answers could be grounded in uploaded company documents. Second, I added an agent layer that can choose between answering from documents or calling support tools like refund or ticket workflows. Third, I added a lightweight logistic regression confidence model that uses retrieval quality and query length to decide when the system should escalate instead of answering confidently. The current version is a prototype, so the support tools are mocked, but the architecture is intentionally designed to mirror a real support automation system."
