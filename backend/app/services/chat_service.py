from __future__ import annotations

import time
from uuid import uuid4

from app.models.schemas import CompareResult, CountType, QueryRecord, TokenMeasurement
from app.services.graph_retrieval_service import GraphRetrievalService
from app.services.llm.base import LLMConfigurationError, LLMProvider
from app.services.retrieval_service import RetrievalService
from app.services.storage import LocalStorage
from app.services.token_service import TokenService


class ChatService:
    def __init__(
        self,
        storage: LocalStorage,
        retrieval_service: RetrievalService,
        graph_retrieval_service: GraphRetrievalService,
        token_service: TokenService,
        llm_provider: LLMProvider,
    ) -> None:
        self.storage = storage
        self.retrieval_service = retrieval_service
        self.graph_retrieval_service = graph_retrieval_service
        self.token_service = token_service
        self.llm_provider = llm_provider

    def standard_qa(self, repo_id: str, query: str, session_id: str | None = None) -> QueryRecord:
        if self.storage.load_repo_metadata(repo_id) is None:
            raise ValueError("Repo not found.")
        session_id = session_id or uuid4().hex
        query_id = uuid4().hex
        started = time.perf_counter()
        retrieval = self.retrieval_service.build_context(repo_id, query)
        prompt = self._standard_prompt(query, retrieval.context)
        token_usage = {
            "chunked_basic_retrieval_context": retrieval.token_measurement,
            "llm_prompt_tokens": self._prompt_measurement(prompt),
        }
        answer = ""
        error = None
        status = "completed"
        try:
            llm_response = self.llm_provider.generate_answer(prompt)
            answer = llm_response.text
            token_usage["llm_prompt_tokens"] = llm_response.prompt_tokens
            token_usage["llm_response_tokens"] = llm_response.response_tokens
            token_usage["total_per_query_tokens"] = llm_response.total_tokens
        except (LLMConfigurationError, RuntimeError) as exc:
            status = "failed"
            error = str(exc)
            token_usage["llm_response_tokens"] = TokenMeasurement(
                stage="llm_response_tokens",
                tokens=0,
                count_type=CountType.exact,
                provider="gemini",
                notes="No response generated.",
            )
            token_usage["total_per_query_tokens"] = TokenMeasurement(
                stage="total_per_query_tokens",
                tokens=token_usage["llm_prompt_tokens"].tokens,
                count_type=token_usage["llm_prompt_tokens"].count_type,
                provider="gemini",
                notes="Query failed before response generation.",
            )
        record = QueryRecord(
            query_id=query_id,
            repo_id=repo_id,
            session_id=session_id,
            mode="standard",
            query=query,
            status=status,
            answer=answer,
            error=error,
            source_snippets=retrieval.snippets,
            token_usage=token_usage,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        self.storage.save_query(record)
        self.storage.append_log(repo_id, "chat-standard", "info", f"Query {query_id} finished with status {status}.")
        return record

    def graph_optimized_qa(self, repo_id: str, query: str, session_id: str | None = None) -> QueryRecord:
        if self.storage.load_repo_metadata(repo_id) is None:
            raise ValueError("Repo not found.")
        session_id = session_id or uuid4().hex
        query_id = uuid4().hex
        started = time.perf_counter()
        graph_context = self.graph_retrieval_service.build_context(repo_id, query)
        prompt = self._graph_prompt(query, graph_context.context)
        token_usage = {
            "codegraph_graphify_optimized_context": graph_context.token_measurement,
            "llm_prompt_tokens": self._prompt_measurement(prompt),
        }
        answer = ""
        error = None
        status = "completed"
        try:
            llm_response = self.llm_provider.generate_answer(prompt)
            answer = llm_response.text
            token_usage["llm_prompt_tokens"] = llm_response.prompt_tokens
            token_usage["llm_response_tokens"] = llm_response.response_tokens
            token_usage["total_per_query_tokens"] = llm_response.total_tokens
        except (LLMConfigurationError, RuntimeError) as exc:
            status = "failed"
            error = str(exc)
            token_usage["llm_response_tokens"] = TokenMeasurement(
                stage="llm_response_tokens",
                tokens=0,
                count_type=CountType.exact,
                provider="gemini",
                notes="No response generated.",
            )
            token_usage["total_per_query_tokens"] = TokenMeasurement(
                stage="total_per_query_tokens",
                tokens=token_usage["llm_prompt_tokens"].tokens,
                count_type=token_usage["llm_prompt_tokens"].count_type,
                provider="gemini",
                notes="Query failed before response generation.",
            )
        record = QueryRecord(
            query_id=query_id,
            repo_id=repo_id,
            session_id=session_id,
            mode="graph_optimized",
            query=query,
            status=status,
            answer=answer,
            error=error,
            source_snippets=graph_context.snippets,
            selected_nodes=graph_context.selected_nodes,
            selected_edges=graph_context.selected_edges,
            token_usage=token_usage,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        self.storage.save_query(record)
        self.storage.append_log(repo_id, "chat-graph", "info", f"Query {query_id} finished with status {status}.")
        return record

    def compare(self, repo_id: str, query: str, session_id: str | None = None) -> CompareResult:
        session_id = session_id or uuid4().hex
        standard = self.standard_qa(repo_id, query, session_id=session_id)
        optimized = self.graph_optimized_qa(repo_id, query, session_id=session_id)
        baseline_tokens = standard.token_usage.get("chunked_basic_retrieval_context")
        optimized_tokens = optimized.token_usage.get("codegraph_graphify_optimized_context")
        baseline_count = baseline_tokens.tokens if baseline_tokens else 0
        optimized_count = optimized_tokens.tokens if optimized_tokens else 0
        saved = baseline_count - optimized_count
        percent = (saved / baseline_count * 100) if baseline_count else 0
        return CompareResult(
            repo_id=repo_id,
            session_id=session_id,
            query=query,
            standard=standard,
            graph_optimized=optimized,
            token_savings={
                "baseline_context_tokens": baseline_count,
                "optimized_context_tokens": optimized_count,
                "saved_context_tokens": saved,
                "saved_percent": round(percent, 2),
                "count_type": "estimated",
            },
            latency_delta_ms=optimized.latency_ms - standard.latency_ms,
        )

    def _prompt_measurement(self, prompt: str) -> TokenMeasurement:
        try:
            return self.llm_provider.count_tokens(prompt, "llm_prompt_tokens")
        except LLMConfigurationError as exc:
            return self.token_service.measure_estimated("llm_prompt_tokens", prompt, notes=str(exc))

    def _standard_prompt(self, query: str, context: str) -> str:
        return (
            "You are a repository QA assistant. Answer only from the provided source context. "
            "Cite file paths and line ranges when possible. If the context is insufficient, say so.\n\n"
            f"Question:\n{query}\n\n"
            f"Source context:\n{context}"
        )

    def _graph_prompt(self, query: str, context: str) -> str:
        return (
            "You are a graph-aware repository QA assistant. Use the selected CodeGraph and Graphify nodes, "
            "relationships, and snippets to answer. Prefer precise relationships over broad guesses. "
            "Cite files, lines, and relevant graph nodes. If the graph context is insufficient, say so.\n\n"
            f"Question:\n{query}\n\n"
            f"Optimized graph context:\n{context}"
        )

