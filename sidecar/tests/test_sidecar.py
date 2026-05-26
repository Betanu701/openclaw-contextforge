from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from contextforge_sidecar.app import Settings, _chunk_text_lossless, create_app


def make_client(
    tmp_path: Path,
    max_context_tokens: int = 512,
    max_node_tokens: int = 768,
) -> TestClient:
    app = create_app(
        Settings(
            db_path=str(tmp_path / "contextforge.db"),
            max_context_tokens=max_context_tokens,
            max_node_tokens=max_node_tokens,
        )
    )
    return TestClient(app)


def namespace(value: str) -> dict[str, str]:
    return {"namespace": value, "sessionId": value.rsplit("/", 1)[-1]}


def test_remember_and_recall_round_trip(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    stored = client.post(
        "/remember",
        json={
            "namespace": namespace("openclaw/test/session-a"),
            "text": "The hidden project codename is BLUE-HERON-773.",
            "title": "Project codename",
            "category": "benchmark",
        },
    )
    assert stored.status_code == 200
    memory_id = stored.json()["id"]
    assert memory_id.startswith("openclaw/test/session-a/")

    recalled = client.post(
        "/recall",
        json={
            "namespace": namespace("openclaw/test/session-a"),
            "query": "What is the hidden project codename?",
            "category": "benchmark",
        },
    )
    assert recalled.status_code == 200
    payload = recalled.json()
    assert "BLUE-HERON-773" in payload["context"]
    assert payload["sources"][0]["id"] == memory_id


def test_namespaces_are_isolated(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    for ns, code in [
        ("openclaw/user-a/session", "ALPHA-111"),
        ("openclaw/user-b/session", "BETA-222"),
    ]:
        response = client.post(
            "/remember",
            json={
                "namespace": namespace(ns),
                "text": f"The access code is {code}.",
                "title": "Access code",
                "category": "benchmark",
            },
        )
        assert response.status_code == 200

    recalled = client.post(
        "/recall",
        json={
            "namespace": namespace("openclaw/user-a/session"),
            "query": "What is the access code?",
            "category": "benchmark",
        },
    )
    assert recalled.status_code == 200
    context = recalled.json()["context"]
    assert "ALPHA-111" in context
    assert "BETA-222" not in context


def test_forget_deletes_only_requested_namespace(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    stored = client.post(
        "/remember",
        json={
            "namespace": namespace("openclaw/test/session-delete"),
            "text": "Delete marker is REMOVE-999.",
            "title": "Delete marker",
            "category": "benchmark",
        },
    )
    assert stored.status_code == 200
    memory_id = stored.json()["id"]

    deleted = client.post(
        "/forget",
        json={"namespace": namespace("openclaw/test/session-delete"), "memoryId": memory_id},
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] == [memory_id]

    recalled = client.post(
        "/recall",
        json={
            "namespace": namespace("openclaw/test/session-delete"),
            "query": "What is the delete marker?",
            "category": "benchmark",
        },
    )
    assert recalled.status_code == 200
    assert recalled.json()["sources"] == []


def test_ingest_chunks_large_text_and_recall_stays_bounded(tmp_path: Path) -> None:
    client = make_client(tmp_path, max_context_tokens=180, max_node_tokens=80)
    filler = "Background telemetry without hidden values stays intentionally mundane.\n" * 40
    text = (
        filler
        + "Unique marker needlechunk42 carries answer ORCHID-424242 for retrieval.\n"
        + filler
    )

    ingested = client.post(
        "/ingest",
        json={
            "namespace": namespace("openclaw/test/chunking"),
            "text": text,
            "title": "Large chunked document",
            "category": "benchmark",
        },
    )
    assert ingested.status_code == 200
    ingest_payload = ingested.json()
    assert ingest_payload["count"] > 1
    assert all("/chunk-" in memory_id for memory_id in ingest_payload["ids"])

    recalled = client.post(
        "/recall",
        json={
            "namespace": namespace("openclaw/test/chunking"),
            "query": "What answer does marker needlechunk42 carry?",
            "category": "benchmark",
            "maxTokens": 180,
        },
    )
    assert recalled.status_code == 200
    payload = recalled.json()
    assert "ORCHID-424242" in payload["context"]
    assert payload["totalTokens"] <= 180


def test_lossless_chunking_preserves_text_near_sentence_boundaries() -> None:
    text = ("a" * 1790) + ". " + ("x" * 400) + "DO-NOT-DROP-MARKER" + ("y" * 1200)
    chunks = _chunk_text_lossless(text, max_tokens=768, overlap=64)
    assert "DO-NOT-DROP-MARKER" in "\n".join(chunks)


def test_recall_phrase_scan_finds_official_niah_style_answer(tmp_path: Path) -> None:
    client = make_client(tmp_path, max_context_tokens=220, max_node_tokens=80)
    filler = (
        "San Francisco archive filler says this thing is best treated as unrelated. "
        "It has no Dolores Park sandwich answer.\n"
    ) * 80
    needle = (
        "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park "
        "on a sunny day.\n"
    )
    ingested = client.post(
        "/ingest",
        json={
            "namespace": namespace("openclaw/test/official-niah"),
            "text": filler + needle + filler,
            "title": "Official NIAH style document",
            "category": "benchmark-official-niah",
        },
    )
    assert ingested.status_code == 200

    recalled = client.post(
        "/recall",
        json={
            "namespace": namespace("openclaw/test/official-niah"),
            "query": "What is the best thing to do in San Francisco?",
            "category": "benchmark-official-niah",
            "maxTokens": 220,
            "limit": 2,
        },
    )
    assert recalled.status_code == 200
    payload = recalled.json()
    assert "eat a sandwich and sit in Dolores Park on a sunny day" in payload["context"]
    assert payload["totalTokens"] <= 220
