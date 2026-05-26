from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from contextforge.index import MemoryIndex, SearchResult
from contextforge.loader import ProactiveLoader
from contextforge.tree import KnowledgeTree, _SCHEMA
from contextforge.utils import chunk_text, estimate_tokens


DEFAULT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".rst",
    ".csv",
    ".xml",
    ".toml",
    ".cfg",
    ".ini",
}


@dataclass(frozen=True)
class Settings:
    db_path: str = "/data/contextforge.db"
    max_context_tokens: int = 4096
    max_node_tokens: int = 768
    ingest_root: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=os.environ.get("CONTEXTFORGE_DB_PATH", "/data/contextforge.db"),
            max_context_tokens=int(os.environ.get("CONTEXTFORGE_MAX_CONTEXT_TOKENS", "4096")),
            max_node_tokens=int(os.environ.get("CONTEXTFORGE_MAX_NODE_TOKENS", "768")),
            ingest_root=os.environ.get("CONTEXTFORGE_INGEST_ROOT") or None,
        )


class Namespace(BaseModel):
    namespace: str
    sessionId: str | None = None
    sessionKey: str | None = None
    agentId: str | None = None
    channelId: str | None = None
    userId: str | None = None
    workspaceDir: str | None = None


class Source(BaseModel):
    id: str
    path: str
    title: str
    category: str
    score: float
    tokens: int
    matchedTerms: list[str] = Field(default_factory=list)


class RecallRequest(BaseModel):
    namespace: Namespace
    query: str
    conversationContext: str | None = None
    category: str | None = None
    maxTokens: int | None = None
    limit: int | None = None


class RecallResponse(BaseModel):
    context: str
    sources: list[Source]
    totalTokens: int
    latencyMs: int


class RememberRequest(BaseModel):
    namespace: Namespace
    text: str
    title: str | None = None
    category: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RememberResponse(BaseModel):
    id: str
    path: str
    title: str
    category: str
    tokens: int


class IngestRequest(BaseModel):
    namespace: Namespace
    text: str | None = None
    path: str | None = None
    title: str | None = None
    category: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    count: int
    ids: list[str]


class ForgetRequest(BaseModel):
    namespace: Namespace
    memoryId: str | None = None
    query: str | None = None
    confirmTopMatch: bool | None = None
    limit: int | None = None


class ForgetResponse(BaseModel):
    deleted: list[str]
    candidates: list[Source]


class StatsResponse(BaseModel):
    dbPath: str
    namespace: str | None = None
    totalNodes: int
    indexedNodes: int
    indexedTerms: int
    categories: dict[str, int]
    cache: dict[str, Any]


def _safe_segment(value: str, fallback: str = "item") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._:-]+", "_", value.strip()).strip("._-")
    return normalized or fallback


def _namespace_path(namespace: Namespace | str) -> str:
    raw = namespace.namespace if isinstance(namespace, Namespace) else namespace
    parts = [_safe_segment(part, "default") for part in raw.split("/") if part.strip()]
    return "/".join(parts) or "default"


def _slug(value: str, fallback: str = "memory") -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or fallback


def _title_from_text(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "Memory")
    return first_line[:80]


class SidecarKnowledgeTree(KnowledgeTree):
    def open(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()


class ContextForgeStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.tree = SidecarKnowledgeTree(db_path=settings.db_path)
        self.tree.open()
        self.index = MemoryIndex()
        self.index.build_from_tree(self.tree)
        self.loader = ProactiveLoader(
            tree=self.tree,
            index=self.index,
            max_context_tokens=settings.max_context_tokens,
        )
        self._lock = RLock()

    def close(self) -> None:
        self.tree.close()

    def remember(
        self,
        namespace: Namespace,
        text: str,
        title: str | None = None,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RememberResponse:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("text must not be empty")
        clean_title = (title or _title_from_text(clean_text)).strip()[:120]
        clean_category = _safe_segment(category or "openclaw", "openclaw")
        memory_metadata = self._metadata(namespace, metadata)
        return self._store_node(namespace, clean_text, clean_title, clean_category, memory_metadata)

    def _metadata(self, namespace: Namespace, metadata: dict[str, Any] | None) -> dict[str, Any]:
        return {
            **(metadata or {}),
            "namespace": namespace.namespace,
            "sessionId": namespace.sessionId,
            "sessionKey": namespace.sessionKey,
            "agentId": namespace.agentId,
            "channelId": namespace.channelId,
            "userId": namespace.userId,
            "workspaceDir": namespace.workspaceDir,
        }

    def _store_node(
        self,
        namespace: Namespace,
        text: str,
        title: str,
        category: str,
        metadata: dict[str, Any],
        path: str | None = None,
        rebuild: bool = True,
    ) -> RememberResponse:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        node_path = path or f"{_namespace_path(namespace)}/{category}/{_slug(title)}-{digest}"
        with self._lock:
            node = self.tree.add(
                path=node_path,
                title=title,
                content=text,
                category=category,
                metadata=metadata,
            )
            if rebuild:
                self.index.build_from_tree(self.tree)
                self.loader.invalidate_cache()
            return RememberResponse(
                id=node.path,
                path=node.path,
                title=node.title,
                category=node.category,
                tokens=node.token_estimate,
            )

    def _ingest_text_document(
        self,
        namespace: Namespace,
        text: str,
        title: str | None,
        category: str,
        metadata: dict[str, Any],
    ) -> list[str]:
        clean_text = text.strip()
        if not clean_text:
            return []
        clean_title = (title or _title_from_text(clean_text)).strip()[:120]
        clean_category = _safe_segment(category, "openclaw")
        chunks = chunk_text(clean_text, max_tokens=self.settings.max_node_tokens, overlap=64)
        if len(chunks) <= 1:
            return [
                self._store_node(
                    namespace,
                    clean_text,
                    clean_title,
                    clean_category,
                    self._metadata(namespace, metadata),
                ).id
            ]

        document_id = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:12]
        base_path = f"{_namespace_path(namespace)}/{clean_category}/{_slug(clean_title)}-{document_id}"
        ids: list[str] = []
        for index, chunk in enumerate(chunks):
            chunk_title = f"{clean_title} (chunk {index + 1}/{len(chunks)})"
            chunk_path = f"{base_path}/chunk-{index + 1:04d}"
            stored = self._store_node(
                namespace,
                chunk,
                chunk_title,
                clean_category,
                self._metadata(
                    namespace,
                    {
                        **metadata,
                        "documentId": base_path,
                        "documentTitle": clean_title,
                        "chunkIndex": index,
                        "chunkCount": len(chunks),
                    },
                ),
                path=chunk_path,
                rebuild=False,
            )
            ids.append(stored.id)
        with self._lock:
            self.index.build_from_tree(self.tree)
            self.loader.invalidate_cache()
        return ids

    def ingest(self, request: IngestRequest) -> IngestResponse:
        ids: list[str] = []
        if request.text:
            ids.extend(
                self._ingest_text_document(
                    request.namespace,
                    request.text,
                    title=request.title,
                    category=request.category or "openclaw",
                    metadata=request.metadata,
                )
            )
        if request.path:
            ids.extend(
                self._ingest_path(
                    request.namespace,
                    request.path,
                    category=request.category or "openclaw",
                    metadata=request.metadata,
                )
            )
        if not ids:
            raise ValueError("provide text or path to ingest")
        return IngestResponse(count=len(ids), ids=ids)

    def _resolve_ingest_path(self, raw_path: str) -> Path:
        if not self.settings.ingest_root:
            raise PermissionError("path ingestion requires CONTEXTFORGE_INGEST_ROOT")
        root = Path(self.settings.ingest_root).resolve()
        candidate = Path(raw_path).expanduser().resolve()
        if candidate != root and root not in candidate.parents:
            raise PermissionError(f"path is outside CONTEXTFORGE_INGEST_ROOT: {candidate}")
        if not candidate.exists():
            raise FileNotFoundError(str(candidate))
        return candidate

    def _ingest_path(
        self,
        namespace: Namespace,
        raw_path: str,
        category: str,
        metadata: dict[str, Any],
    ) -> list[str]:
        path = self._resolve_ingest_path(raw_path)
        files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
        ids: list[str] = []
        for file_path in files:
            if file_path.suffix.lower() not in DEFAULT_EXTENSIONS:
                continue
            content = file_path.read_text(encoding="utf-8", errors="replace")
            relative_title = str(file_path.relative_to(path.parent if path.is_file() else path))
            ids.extend(
                self._ingest_text_document(
                    namespace,
                    content,
                    title=relative_title,
                    category=category,
                    metadata={**metadata, "sourcePath": str(file_path)},
                )
            )
        return ids

    def recall(self, request: RecallRequest) -> RecallResponse:
        started = time.perf_counter()
        max_tokens = max(1, request.maxTokens or self.settings.max_context_tokens)
        limit = max(1, min(request.limit or 8, 50))
        namespace_prefix = f"{_namespace_path(request.namespace)}/"
        combined_query = " ".join(
            part.strip()
            for part in [request.query, request.conversationContext or ""]
            if part and part.strip()
        )
        if not combined_query:
            return RecallResponse(context="", sources=[], totalTokens=0, latencyMs=0)

        with self._lock:
            results = self.index.search(combined_query, top_k=max(self.index.num_docs, limit * 20, 100))
            selected = self._select_results(results, namespace_prefix, request.category, max_tokens, limit)
            context, sources, tokens = self._assemble(selected, max_tokens)

        return RecallResponse(
            context=context,
            sources=sources,
            totalTokens=tokens,
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    def _select_results(
        self,
        results: list[SearchResult],
        namespace_prefix: str,
        category: str | None,
        max_tokens: int,
        limit: int,
    ) -> list[SearchResult]:
        selected: list[SearchResult] = []
        total_tokens = 0
        for result in results:
            if not result.path.startswith(namespace_prefix):
                continue
            if category and result.category != category:
                continue
            node = self.tree.get(result.path)
            if not node:
                continue
            entry_tokens = node.token_estimate + estimate_tokens(f"\n### {node.title} [{node.path}]")
            next_total = total_tokens + entry_tokens
            if next_total > max_tokens:
                continue
            selected.append(result)
            total_tokens = next_total
            if len(selected) >= limit:
                break
        return selected

    def _assemble(self, results: list[SearchResult], max_tokens: int) -> tuple[str, list[Source], int]:
        if not results:
            return "", [], 0
        parts = ["## Relevant ContextForge Memory"]
        sources: list[Source] = []
        for result in results:
            node = self.tree.get(result.path)
            if not node:
                continue
            candidate_parts = [*parts, f"\n### {node.title} [{node.path}]", node.content]
            if estimate_tokens("\n".join(candidate_parts)) > max_tokens:
                continue
            parts = candidate_parts
            sources.append(
                Source(
                    id=node.path,
                    path=node.path,
                    title=node.title,
                    category=node.category,
                    score=result.score,
                    tokens=node.token_estimate,
                    matchedTerms=result.matched_terms,
                )
            )
        context = "\n".join(parts) if sources else ""
        return context, sources, estimate_tokens(context)

    def forget(self, request: ForgetRequest) -> ForgetResponse:
        namespace_prefix = f"{_namespace_path(request.namespace)}/"
        deleted: list[str] = []
        candidates: list[Source] = []
        with self._lock:
            if request.memoryId:
                if not request.memoryId.startswith(namespace_prefix):
                    raise PermissionError("memory id is outside the requested namespace")
                if self.tree.remove(request.memoryId):
                    deleted.append(request.memoryId)
                    self.index.build_from_tree(self.tree)
                    self.loader.invalidate_cache()
                return ForgetResponse(deleted=deleted, candidates=[])

            if not request.query:
                raise ValueError("provide memoryId or query")

            recall = self.recall(
                RecallRequest(
                    namespace=request.namespace,
                    query=request.query,
                    limit=request.limit or 5,
                    maxTokens=self.settings.max_context_tokens,
                )
            )
            candidates = recall.sources
            if request.confirmTopMatch and candidates:
                top = candidates[0].id
                if self.tree.remove(top):
                    deleted.append(top)
                    self.index.build_from_tree(self.tree)
                    self.loader.invalidate_cache()
                candidates = []
        return ForgetResponse(deleted=deleted, candidates=candidates)

    def stats(self, namespace: str | None = None) -> StatsResponse:
        with self._lock:
            if namespace:
                prefix = f"{_namespace_path(namespace)}/%"
                rows = self.tree.conn.execute(
                    "SELECT category, COUNT(*) FROM knowledge_nodes WHERE path LIKE ? GROUP BY category",
                    (prefix,),
                ).fetchall()
                total = sum(int(row[1]) for row in rows)
            else:
                rows = self.tree.conn.execute(
                    "SELECT category, COUNT(*) FROM knowledge_nodes GROUP BY category"
                ).fetchall()
                total = self.tree.total_nodes()
            return StatsResponse(
                dbPath=self.settings.db_path,
                namespace=namespace,
                totalNodes=total,
                indexedNodes=self.index.num_docs,
                indexedTerms=self.index.num_terms,
                categories={str(row[0]): int(row[1]) for row in rows},
                cache=self.loader.cache_stats(),
            )


def create_app(settings: Settings | None = None) -> FastAPI:
    store = ContextForgeStore(settings or Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="ContextForge OpenClaw Sidecar", version="0.1.0", lifespan=lifespan)
    app.state.store = store

    @app.get("/healthz", response_model=StatsResponse)
    async def healthz() -> StatsResponse:
        return store.stats()

    @app.post("/recall", response_model=RecallResponse)
    async def recall(request: RecallRequest) -> RecallResponse:
        try:
            return store.recall(request)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/remember", response_model=RememberResponse)
    async def remember(request: RememberRequest) -> RememberResponse:
        try:
            return store.remember(
                request.namespace,
                request.text,
                title=request.title,
                category=request.category,
                metadata=request.metadata,
            )
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest) -> IngestResponse:
        try:
            return store.ingest(request)
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/forget", response_model=ForgetResponse)
    async def forget(request: ForgetRequest) -> ForgetResponse:
        try:
            return store.forget(request)
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/stats", response_model=StatsResponse)
    async def stats(namespace: str | None = Query(default=None)) -> StatsResponse:
        return store.stats(namespace)

    return app


def main() -> None:
    uvicorn.run(
        "contextforge_sidecar.app:create_app",
        factory=True,
        host=os.environ.get("CONTEXTFORGE_HOST", "0.0.0.0"),
        port=int(os.environ.get("CONTEXTFORGE_PORT", "8765")),
    )
