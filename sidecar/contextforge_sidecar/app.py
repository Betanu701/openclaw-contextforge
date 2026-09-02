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
from contextforge.providers import LLMProvider, get_provider
from contextforge.session import Session, SessionStore, _SCHEMA as SESSION_SCHEMA
from contextforge.tree import KnowledgeTree, _SCHEMA as TREE_SCHEMA
from contextforge.utils import estimate_tokens, extract_keywords


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

PERMANENT_CONTEXT_CATEGORY = "_permanent_context"

QUESTION_LEADERS = {
    "what",
    "which",
    "where",
    "when",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "the",
    "a",
    "an",
}


@dataclass(frozen=True)
class Settings:
    db_path: str = "/data/contextforge.db"
    max_context_tokens: int = 4096
    max_node_tokens: int = 768
    ingest_root: str | None = None
    llm_provider: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str = ""
    system_prompt: str = "You are a helpful assistant with access to ContextForge memory."

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=os.environ.get("CONTEXTFORGE_DB_PATH", "/data/contextforge.db"),
            max_context_tokens=int(os.environ.get("CONTEXTFORGE_MAX_CONTEXT_TOKENS", "4096")),
            max_node_tokens=int(os.environ.get("CONTEXTFORGE_MAX_NODE_TOKENS", "768")),
            ingest_root=os.environ.get("CONTEXTFORGE_INGEST_ROOT") or None,
            llm_provider=os.environ.get("CONTEXTFORGE_LLM_PROVIDER") or None,
            llm_base_url=os.environ.get("CONTEXTFORGE_LLM_BASE_URL") or None,
            llm_model=os.environ.get("CONTEXTFORGE_LLM_MODEL") or None,
            llm_api_key=os.environ.get("CONTEXTFORGE_LLM_API_KEY", ""),
            system_prompt=os.environ.get(
                "CONTEXTFORGE_SYSTEM_PROMPT",
                "You are a helpful assistant with access to ContextForge memory.",
            ),
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


class ContextRequest(BaseModel):
    namespace: Namespace
    query: str
    conversationContext: str | None = None
    category: str | None = None
    maxTokens: int | None = None
    limit: int | None = None
    includePermanent: bool = True


class ContextResponse(BaseModel):
    context: str
    sources: list[Source]
    totalTokens: int
    latencyMs: int
    permanentTokens: int = 0
    branchPaths: list[str] = Field(default_factory=list)


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


class PermanentContextRequest(BaseModel):
    namespace: Namespace
    text: str
    title: str | None = None


class PermanentContextResponse(BaseModel):
    id: str
    tokens: int


class SessionRequest(BaseModel):
    namespace: Namespace
    sessionId: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionMessageRequest(BaseModel):
    namespace: Namespace
    role: str
    content: str
    sessionId: str | None = None


class SessionResponse(BaseModel):
    id: str
    resumed: bool
    messageCount: int
    totalTokens: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionsResponse(BaseModel):
    sessions: list[SessionResponse]


class ChatRequest(BaseModel):
    namespace: Namespace
    message: str
    sessionId: str | None = None
    category: str | None = None
    maxTokens: int | None = None
    limit: int | None = None
    modelKwargs: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    response: str
    sessionId: str
    context: ContextResponse
    latencyMs: int


class AnalyzeRequest(ChatRequest):
    maxPasses: int = 4


class AnalyzeResponse(BaseModel):
    response: str
    sessionId: str
    contexts: list[ContextResponse]
    latencyMs: int


class StatsResponse(BaseModel):
    dbPath: str
    namespace: str | None = None
    totalNodes: int
    indexedNodes: int
    indexedTerms: int
    categories: dict[str, int]
    cache: dict[str, Any]
    sessions: int = 0
    permanentContextTokens: int = 0
    modelConfigured: bool = False


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


def _normalize_phrase_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _query_phrase_candidates(query: str) -> list[str]:
    normalized = _normalize_phrase_text(query)
    if not normalized:
        return []

    candidates = [normalized]
    words = normalized.split()
    while words and words[0] in QUESTION_LEADERS:
        words = words[1:]
        candidate = " ".join(words)
        if len(candidate.split()) >= 2:
            candidates.append(candidate)

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _chunk_text_lossless(text: str, max_tokens: int = 512, overlap: int = 64) -> list[str]:
    if not text:
        return []

    max_chars = max(1, int(max_tokens * 4.0))
    overlap_chars = max(0, min(int(overlap * 4.0), max_chars - 1))
    chunks: list[str] = []
    start = 0

    while start < len(text):
        target_end = min(len(text), start + max_chars)
        end = target_end
        if target_end < len(text):
            window = text[start:target_end]
            for separator in ("\n\n", "\n", ". ", "! ", "? ", " "):
                split_at = window.rfind(separator)
                if split_at > max_chars // 2:
                    end = start + split_at + len(separator)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)

    return chunks


def _fit_to_token_budget(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    return text[: max_tokens * 4].rstrip()


class SidecarKnowledgeTree(KnowledgeTree):
    def open(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(TREE_SCHEMA)
        self._conn.commit()


class SidecarSessionStore(SessionStore):
    def open(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SESSION_SCHEMA)
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
        self.sessions = SidecarSessionStore(db_path=settings.db_path)
        self.sessions.open()
        self._provider: LLMProvider | None = None
        self._lock = RLock()

    def close(self) -> None:
        self.tree.close()
        self.sessions.close()

    def _get_provider(self) -> LLMProvider:
        if not self.settings.llm_provider:
            raise ValueError(
                "ContextForge sidecar LLM is not configured; set CONTEXTFORGE_LLM_PROVIDER "
                "and related CONTEXTFORGE_LLM_* environment variables to enable chat/analyze"
            )
        if self._provider is None:
            self._provider = get_provider(
                self.settings.llm_provider,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model,
                base_url=self.settings.llm_base_url,
            )
        return self._provider

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

    def set_permanent_context(self, request: PermanentContextRequest) -> PermanentContextResponse:
        clean_text = request.text.strip()
        if not clean_text:
            raise ValueError("text must not be empty")
        title = (request.title or "Permanent Context").strip()[:120]
        path = f"{_namespace_path(request.namespace)}/{PERMANENT_CONTEXT_CATEGORY}/permanent"
        stored = self._store_node(
            request.namespace,
            clean_text,
            title,
            PERMANENT_CONTEXT_CATEGORY,
            self._metadata(request.namespace, {"source": "openclaw_permanent_context"}),
            path=path,
        )
        return PermanentContextResponse(id=stored.id, tokens=stored.tokens)

    def _permanent_context(self, namespace: Namespace, max_tokens: int) -> tuple[str, int]:
        path = f"{_namespace_path(namespace)}/{PERMANENT_CONTEXT_CATEGORY}/permanent"
        node = self.tree.get(path)
        if not node:
            return "", 0
        text = _fit_to_token_budget(node.content, max_tokens)
        return text, estimate_tokens(text)

    def _session_id(self, namespace: Namespace, session_id: str | None = None) -> str:
        explicit = session_id or namespace.sessionId or namespace.sessionKey or namespace.channelId or "default"
        return f"{_namespace_path(namespace)}/sessions/{_safe_segment(explicit, 'default')}"

    def _session_response(self, session: Session, resumed: bool) -> SessionResponse:
        return SessionResponse(
            id=session.id,
            resumed=resumed,
            messageCount=len(session.messages),
            totalTokens=session.total_tokens,
            metadata=session.metadata,
        )

    def start_session(self, request: SessionRequest) -> SessionResponse:
        session_id = self._session_id(request.namespace, request.sessionId)
        with self._lock:
            existing = self.sessions.load_session(session_id)
            if existing:
                return self._session_response(existing, resumed=True)
            session = self.sessions.create_session(
                session_id=session_id,
                metadata=self._metadata(request.namespace, request.metadata),
            )
            return self._session_response(session, resumed=False)

    def list_sessions(self, namespace: Namespace) -> SessionsResponse:
        prefix = f"{_namespace_path(namespace)}/sessions/"
        with self._lock:
            sessions = []
            for entry in self.sessions.list_sessions():
                session_id = str(entry["id"])
                if not session_id.startswith(prefix):
                    continue
                session = self.sessions.load_session(session_id)
                if session:
                    sessions.append(self._session_response(session, resumed=True))
            return SessionsResponse(sessions=sessions)

    def add_session_message(self, request: SessionMessageRequest) -> SessionResponse:
        role = request.role.strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("role must be one of: system, user, assistant, tool")
        content = request.content.strip()
        if not content:
            raise ValueError("content must not be empty")
        session = self._ensure_session(request.namespace, request.sessionId)
        with self._lock:
            self.sessions.add_message(session.id, role, content)
            updated = self.sessions.load_session(session.id)
            if not updated:
                raise ValueError(f"session not found after message add: {session.id}")
            return self._session_response(updated, resumed=True)

    def _ensure_session(self, namespace: Namespace, session_id: str | None = None) -> Session:
        resolved = self._session_id(namespace, session_id)
        with self._lock:
            existing = self.sessions.load_session(resolved)
            if existing:
                return existing
            return self.sessions.create_session(
                session_id=resolved,
                metadata=self._metadata(namespace, {"source": "openclaw_contextforge"}),
            )

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
        chunks = _chunk_text_lossless(clean_text, max_tokens=self.settings.max_node_tokens, overlap=64)
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
            results = self._augment_results_with_phrase_scan(
                results,
                combined_query,
                namespace_prefix,
                request.category,
            )
            selected = self._select_results(results, namespace_prefix, request.category, max_tokens, limit)
            context, sources, tokens = self._assemble(selected, max_tokens)

        return RecallResponse(
            context=context,
            sources=sources,
            totalTokens=tokens,
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    def context(self, request: ContextRequest) -> ContextResponse:
        started = time.perf_counter()
        max_tokens = max(1, request.maxTokens or self.settings.max_context_tokens)
        permanent = ""
        permanent_tokens = 0
        if request.includePermanent:
            permanent, permanent_tokens = self._permanent_context(request.namespace, max_tokens)

        recall_budget = max(1, max_tokens - permanent_tokens)
        recall = self.recall(
            RecallRequest(
                namespace=request.namespace,
                query=request.query,
                conversationContext=request.conversationContext,
                category=request.category,
                maxTokens=recall_budget,
                limit=request.limit,
            )
        )

        parts: list[str] = []
        if permanent:
            parts.append("## Permanent ContextForge Context\n" + permanent)
        if recall.context:
            parts.append(recall.context)
        context = "\n\n".join(parts)

        return ContextResponse(
            context=context,
            sources=recall.sources,
            totalTokens=estimate_tokens(context),
            latencyMs=int((time.perf_counter() - started) * 1000),
            permanentTokens=permanent_tokens,
            branchPaths=[source.id for source in recall.sources],
        )

    def _recent_session_context(self, session_id: str, max_turns: int = 6) -> str:
        with self._lock:
            session = self.sessions.load_session(session_id)
            if not session:
                return ""
            recent = session.messages[-max_turns:]
        return " ".join(
            str(message.get("content", ""))
            for message in recent
            if message.get("role") != "system"
        )

    def _build_llm_messages(
        self,
        session_id: str,
        user_message: str,
        context: ContextResponse,
        *,
        task_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        system_parts = [self.settings.system_prompt]
        if task_prompt:
            system_parts.append(task_prompt)
        if context.context:
            system_parts.append(context.context)
        messages = [{"role": "system", "content": "\n\n".join(system_parts)}]

        with self._lock:
            session = self.sessions.load_session(session_id)
            history = list(session.messages[-12:]) if session else []
        for message in history:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role in {"user", "assistant", "tool"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})
        return messages

    async def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        provider = self._get_provider()
        session = self._ensure_session(request.namespace, request.sessionId)
        recent_context = self._recent_session_context(session.id)
        context = self.context(
            ContextRequest(
                namespace=request.namespace,
                query=request.message,
                conversationContext=recent_context,
                category=request.category,
                maxTokens=request.maxTokens,
                limit=request.limit,
            )
        )
        messages = self._build_llm_messages(session.id, request.message, context)
        response = await provider.chat(messages, **request.modelKwargs)

        with self._lock:
            self.sessions.add_message(session.id, "user", request.message)
            self.sessions.add_message(session.id, "assistant", response)

        return ChatResponse(
            response=response,
            sessionId=session.id,
            context=context,
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    def _matching_categories(
        self,
        namespace: Namespace,
        query: str,
        category: str | None,
        max_passes: int,
    ) -> list[str]:
        if category:
            return [category]

        namespace_prefix = f"{_namespace_path(namespace)}/"
        with self._lock:
            results = self.index.search(query, top_k=max(self.index.num_docs, max_passes * 20, 100))
            results = self._augment_results_with_phrase_scan(
                results,
                query,
                namespace_prefix,
                None,
            )

        categories: list[str] = []
        for result in results:
            if not result.path.startswith(namespace_prefix):
                continue
            if result.category == PERMANENT_CONTEXT_CATEGORY:
                continue
            if result.category not in categories:
                categories.append(result.category)
            if len(categories) >= max_passes:
                break
        return categories

    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        started = time.perf_counter()
        provider = self._get_provider()
        session = self._ensure_session(request.namespace, request.sessionId)
        recent_context = self._recent_session_context(session.id)
        max_passes = max(1, min(request.maxPasses, 8))
        categories = self._matching_categories(
            request.namespace,
            request.message,
            request.category,
            max_passes,
        )

        contexts = [
            self.context(
                ContextRequest(
                    namespace=request.namespace,
                    query=request.message,
                    conversationContext=recent_context,
                    category=category,
                    maxTokens=request.maxTokens,
                    limit=request.limit,
                )
            )
            for category in categories
        ]
        if not contexts:
            contexts = [
                self.context(
                    ContextRequest(
                        namespace=request.namespace,
                        query=request.message,
                        conversationContext=recent_context,
                        maxTokens=request.maxTokens,
                        limit=request.limit,
                    )
                )
            ]

        if len(contexts) == 1:
            messages = self._build_llm_messages(
                session.id,
                request.message,
                contexts[0],
                task_prompt="Analyze the question using the supplied ContextForge context.",
            )
            response = await provider.chat(messages, **request.modelKwargs)
        else:
            domain_responses: list[str] = []
            for index, context in enumerate(contexts, start=1):
                messages = self._build_llm_messages(
                    session.id,
                    request.message,
                    context,
                    task_prompt=(
                        f"Pass {index}: analyze the question using only this ContextForge "
                        "knowledge domain. Note agreements, contradictions, and gaps."
                    ),
                )
                domain_responses.append(await provider.chat(messages, **request.modelKwargs))

            synthesis = [
                f"You were asked: {request.message}",
                f"Here are {len(domain_responses)} ContextForge domain analyses:",
            ]
            for index, response in enumerate(domain_responses, start=1):
                synthesis.append(f"--- Domain {index} ---\n{response}")
            synthesis.append(
                "Synthesize one answer. Call out agreements, contradictions, missing evidence, and the most useful next action."
            )
            response = await provider.chat(
                [
                    {"role": "system", "content": self.settings.system_prompt},
                    {"role": "user", "content": "\n\n".join(synthesis)},
                ],
                **request.modelKwargs,
            )

        with self._lock:
            self.sessions.add_message(session.id, "user", request.message)
            self.sessions.add_message(session.id, "assistant", response)

        return AnalyzeResponse(
            response=response,
            sessionId=session.id,
            contexts=contexts,
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    def _augment_results_with_phrase_scan(
        self,
        results: list[SearchResult],
        query: str,
        namespace_prefix: str,
        category: str | None,
    ) -> list[SearchResult]:
        query_terms = extract_keywords(query, top_k=15)
        phrase_candidates = _query_phrase_candidates(query)
        if not query_terms and not phrase_candidates:
            return results

        rows = self.tree.conn.execute(
            """
            SELECT id, path, title, category, content
            FROM knowledge_nodes
            WHERE path LIKE ?
            """,
            (f"{namespace_prefix}%",),
        ).fetchall()
        phrase_results: list[SearchResult] = []
        for node_id, path, title, node_category, content in rows:
            if category and node_category != category:
                continue
            normalized_content = _normalize_phrase_text(str(content))
            content_words = set(normalized_content.split())
            matched_terms = [term for term in query_terms if term in content_words]
            phrase_hits = [
                phrase for phrase in phrase_candidates if phrase and phrase in normalized_content
            ]
            if not matched_terms and not phrase_hits:
                continue

            unique_terms = sorted(set(matched_terms))
            score = float(len(unique_terms) * 25)
            if query_terms and len(unique_terms) == len(set(query_terms)):
                score += 100.0
            score += float(len(phrase_hits) * 500)

            phrase_results.append(
                SearchResult(
                    node_id=int(node_id),
                    path=str(path),
                    title=str(title),
                    category=str(node_category),
                    score=score,
                    matched_terms=unique_terms + phrase_hits,
                )
            )

        if not phrase_results:
            return results

        merged: dict[str, SearchResult] = {result.path: result for result in results}
        for result in phrase_results:
            existing = merged.get(result.path)
            if not existing or result.score > existing.score:
                merged[result.path] = result
        return sorted(merged.values(), key=lambda result: -result.score)

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
                session_count = len(
                    [
                        session
                        for session in self.sessions.list_sessions()
                        if str(session["id"]).startswith(f"{_namespace_path(namespace)}/sessions/")
                    ]
                )
                permanent_path = f"{_namespace_path(namespace)}/{PERMANENT_CONTEXT_CATEGORY}/permanent"
                permanent_node = self.tree.get(permanent_path)
                permanent_tokens = permanent_node.token_estimate if permanent_node else 0
            else:
                rows = self.tree.conn.execute(
                    "SELECT category, COUNT(*) FROM knowledge_nodes GROUP BY category"
                ).fetchall()
                total = self.tree.total_nodes()
                session_count = len(self.sessions.list_sessions())
                permanent_row = self.tree.conn.execute(
                    "SELECT COALESCE(SUM(token_estimate), 0) FROM knowledge_nodes WHERE category = ?",
                    (PERMANENT_CONTEXT_CATEGORY,),
                ).fetchone()
                permanent_tokens = int(permanent_row[0]) if permanent_row else 0
            return StatsResponse(
                dbPath=self.settings.db_path,
                namespace=namespace,
                totalNodes=total,
                indexedNodes=self.index.num_docs,
                indexedTerms=self.index.num_terms,
                categories={str(row[0]): int(row[1]) for row in rows},
                cache=self.loader.cache_stats(),
                sessions=session_count,
                permanentContextTokens=permanent_tokens,
                modelConfigured=bool(self.settings.llm_provider),
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

    @app.post("/context", response_model=ContextResponse)
    async def context(request: ContextRequest) -> ContextResponse:
        try:
            return store.context(request)
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

    @app.post("/permanent-context", response_model=PermanentContextResponse)
    async def permanent_context(request: PermanentContextRequest) -> PermanentContextResponse:
        try:
            return store.set_permanent_context(request)
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

    @app.post("/session/start", response_model=SessionResponse)
    async def session_start(request: SessionRequest) -> SessionResponse:
        try:
            return store.start_session(request)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/session/message", response_model=SessionResponse)
    async def session_message(request: SessionMessageRequest) -> SessionResponse:
        try:
            return store.add_session_message(request)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/sessions/list", response_model=SessionsResponse)
    async def sessions_list(request: SessionRequest) -> SessionsResponse:
        try:
            return store.list_sessions(request.namespace)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            return await store.chat(request)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
        try:
            return await store.analyze(request)
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
