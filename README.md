# ResolveAI

ResolveAI is a backend AI customer support copilot built with FastAPI, LangChain, FAISS, and a lightweight scikit-learn confidence model.

It supports:
- PDF document upload and indexing for retrieval-augmented generation (RAG)
- Agent-based query handling with tool calling for support workflows
- Confidence-based escalation when retrieved context is weak

## Current Scope

This repo currently focuses on the backend. The support tools for refunds, tickets, and order checks are demo actions that return mock responses instead of calling real external systems.

## Architecture

1. Users upload PDF documents through `/upload-doc`.
2. The backend chunks the document, creates embeddings, and stores vectors in FAISS.
3. `/ask` and `/stream-ask` route user questions through a LangChain agent.
4. The RAG tool retrieves the most relevant document chunks.
5. A logistic regression model estimates confidence using retrieval similarity and query length.
6. Low-confidence answers are escalated instead of answered directly.

## Tech Stack

- FastAPI
- LangChain
- OpenAI embeddings and chat model
- FAISS
- scikit-learn
- pytest

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables:

```bash
copy .env.example .env
```

Then fill in `OPENAI_API_KEY`.

4. Train the confidence model:

```bash
python train_model.py
```

5. Start the API:

```bash
uvicorn app.main:app --reload
```

## API Endpoints

- `GET /health` - health check
- `POST /upload-doc` - upload a PDF document for indexing
- `POST /ask` - ask a question
- `GET /stream-ask` - stream an answer

## Tests

Run the backend tests with:

```bash
pytest
```

## Resume-Friendly Summary

Built an AI customer support copilot backend using FastAPI, LangChain, and FAISS to answer questions over uploaded company documents, with an agent workflow for support operations and a logistic regression confidence model for escalation.
