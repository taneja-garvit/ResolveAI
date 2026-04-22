from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.routes import chat, upload
from app.services import agent_service


client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_route_returns_agent_output(monkeypatch):
    monkeypatch.setattr(chat, "run_agent", lambda query: f"answer for {query}")

    response = client.post("/ask", params={"query": "refund status"})

    assert response.status_code == 200
    assert response.json() == {"answer": "answer for refund status"}


def test_stream_ask_returns_stream_body(monkeypatch):
    def fake_stream_agent(query):
        yield f"streamed answer for {query}"

    monkeypatch.setattr(chat, "stream_agent", fake_stream_agent)

    with client.stream("GET", "/stream-ask", params={"query": "hello"}) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert body == "streamed answer for hello"


def test_upload_rejects_non_pdf():
    response = client.post(
        "/upload-doc",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF uploads are supported."


def test_upload_sanitizes_filename_and_processes_document(monkeypatch, tmp_path):
    saved_paths = []

    def fake_process_document(file_path):
        saved_paths.append(Path(file_path))

    monkeypatch.setattr(upload, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(upload, "process_document", fake_process_document)

    response = client.post(
        "/upload-doc",
        files={"file": ("../../company-handbook.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    assert response.status_code == 200
    assert saved_paths
    assert saved_paths[0].parent == tmp_path
    assert ".." not in saved_paths[0].name
    assert saved_paths[0].suffix == ".pdf"


def test_rag_tool_uses_retrieval_scores_for_confidence(monkeypatch):
    captured = {}
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FakeDoc:
        def __init__(self, page_content):
            self.page_content = page_content

    class FakeVectorStore:
        def similarity_search_with_score(self, query, k=3):
            return [
                (FakeDoc("Refunds are processed in 3-5 days."), 0.1),
                (FakeDoc("Tickets can be escalated to human support."), 0.4),
            ]

    class FakeResponse:
        content = "Here is the answer from the knowledge base."

    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, messages):
            return FakeResponse()

    def fake_predict_confidence(similarity_score, query_length):
        captured["similarity_score"] = similarity_score
        captured["query_length"] = query_length
        return 0.92

    monkeypatch.setattr(agent_service, "get_vectorstore", lambda: FakeVectorStore())
    monkeypatch.setattr(agent_service, "predict_confidence", fake_predict_confidence)
    monkeypatch.setattr(agent_service, "ChatOpenAI", FakeChatOpenAI)

    response = agent_service.rag_tool("How long do refunds take?")

    assert "confidence: 0.92" in response
    assert captured["query_length"] == len("How long do refunds take?")
    assert 0.0 < captured["similarity_score"] < 1.0
    assert captured["similarity_score"] != (2 / 3)
