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
        pipeline: Any = None,
    ) -> None:
        self.storage = storage
        self.retrieval_service = retrieval_service
        self.graph_retrieval_service = graph_retrieval_service
        self.token_service = token_service
        self.llm_provider = llm_provider
        
        # Instantiate RectificationService natively
        from app.services.rectification_service import RectificationService
        self.rectification_service = RectificationService(storage, pipeline)

    def apply_rectification(self, repo_id: str, file_path: str, original_code: str, replacement_code: str) -> dict[str, Any]:
        return self.rectification_service.apply_code_fix(repo_id, file_path, original_code, replacement_code)

    def standard_qa(self, repo_id: str, query: str, session_id: str | None = None, limit: int = 8, rectify: bool = False) -> QueryRecord:
        if self.storage.load_repo_metadata(repo_id) is None:
            raise ValueError("Repo not found.")
        session_id = session_id or uuid4().hex
        query_id = uuid4().hex
        started = time.perf_counter()
        retrieval = self.retrieval_service.build_context(repo_id, query, limit=limit)
        prompt = self._standard_prompt(query, retrieval.context, rectify)
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

    def graph_optimized_qa(self, repo_id: str, query: str, session_id: str | None = None, source_selection: str = "merged", max_nodes: int = 8, rectify: bool = False) -> QueryRecord:
        if self.storage.load_repo_metadata(repo_id) is None:
            raise ValueError("Repo not found.")
        session_id = session_id or uuid4().hex
        query_id = uuid4().hex
        started = time.perf_counter()
        graph_context = self.graph_retrieval_service.build_context(repo_id, query, max_nodes=max_nodes, source_selection=source_selection)
        prompt = self._graph_prompt(query, graph_context.context, rectify)
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
            
        # Compute Whole Codebase Baseline (General Chatbot approach: Concatenated whole repository code files)
        try:
            repo_tokens = self._get_total_repo_tokens(repo_id)
            query_prompt = self._standard_prompt(query, "[concatenated_files_placeholder]")
            query_tokens = self._prompt_measurement(query_prompt).tokens
            baseline_total = repo_tokens + query_tokens
        except Exception:
            baseline_total = 10000 # Safety fallback
            
        token_usage["whole_codebase_baseline"] = TokenMeasurement(
            stage="whole_codebase_baseline",
            tokens=baseline_total,
            count_type=CountType.exact if self.token_service._encoding else CountType.estimated,
            notes="Prompt tokens required if sending 100% of codebase files directly to LLM."
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

    def compare(
        self,
        repo_id: str,
        query: str,
        session_id: str | None = None,
        source_selection: str = "merged",
        limit: int = 8,
        max_nodes: int = 8,
    ) -> CompareResult:
        session_id = session_id or uuid4().hex
        standard = self.standard_qa(repo_id, query, session_id=session_id, limit=limit)
        optimized = self.graph_optimized_qa(repo_id, query, session_id=session_id, source_selection=source_selection, max_nodes=max_nodes)
        
        # Get LLM Prompt Tokens (Actual tokens sent to the LLM)
        baseline_prompt_tokens = standard.token_usage.get("llm_prompt_tokens")
        optimized_prompt_tokens = optimized.token_usage.get("llm_prompt_tokens")
        
        baseline_prompt_count = baseline_prompt_tokens.tokens if baseline_prompt_tokens else 0
        optimized_prompt_count = optimized_prompt_tokens.tokens if optimized_prompt_tokens else 0
        
        saved_prompt = baseline_prompt_count - optimized_prompt_count
        percent_prompt = (saved_prompt / baseline_prompt_count * 100) if baseline_prompt_count else 0
        
        # Context tokens (for backward compatibility / secondary check)
        baseline_context = standard.token_usage.get("chunked_basic_retrieval_context")
        optimized_context = optimized.token_usage.get("codegraph_graphify_optimized_context")
        
        baseline_context_count = baseline_context.tokens if baseline_context else 0
        optimized_context_count = optimized_context.tokens if optimized_context else 0
        
        saved_context = baseline_context_count - optimized_context_count
        percent_context = (saved_context / baseline_context_count * 100) if baseline_context_count else 0
        
        return CompareResult(
            repo_id=repo_id,
            session_id=session_id,
            query=query,
            standard=standard,
            graph_optimized=optimized,
            token_savings={
                "baseline_prompt_tokens": baseline_prompt_count,
                "optimized_prompt_tokens": optimized_prompt_count,
                "saved_prompt_tokens": saved_prompt,
                "saved_percent": round(percent_prompt, 2),
                
                # Context tokens
                "baseline_context_tokens": baseline_context_count,
                "optimized_context_tokens": optimized_context_count,
                "saved_context_tokens": saved_context,
                "saved_context_percent": round(percent_context, 2),
                "count_type": "exact",
            },
            latency_delta_ms=optimized.latency_ms - standard.latency_ms,
        )

    def _prompt_measurement(self, prompt: str) -> TokenMeasurement:
        try:
            return self.llm_provider.count_tokens(prompt, "llm_prompt_tokens")
        except LLMConfigurationError as exc:
            return self.token_service.measure_estimated("llm_prompt_tokens", prompt, notes=str(exc))

    def _standard_prompt(self, query: str, context: str, rectify: bool = False) -> str:
        rectify_str = ""
        if rectify:
            rectify_str = (
                "\n\nIMPORTANT: If you identify any bug/error and propose a code change, you MUST wrap the proposed fix exactly in "
                "the following XML structure so the system can apply it automatically:\n"
                "<code_fix>\n"
                "  <filepath>relative/path/to/file.py</filepath>\n"
                "  <original_code>\n"
                "// Exact block of old code to replace (must match precisely including spacing)\n"
                "  </original_code>\n"
                "  <replacement_code>\n"
                "// Exact block of new code to insert\n"
                "  </replacement_code>\n"
                "</code_fix>\n"
                "Make sure that the <original_code> block you target matches the codebase content exactly, character-for-character."
            )
        return (
            "You are a repository QA assistant. Answer only from the provided source context. "
            "Cite file paths and line ranges when possible. If the context is insufficient, say so. "
            "Be extremely concise, direct, and brief in your answer. Avoid verbose explanations."
            f"{rectify_str}\n\n"
            f"Question:\n{query}\n\n"
            f"Source context:\n{context}"
        )

    def _graph_prompt(self, query: str, context: str, rectify: bool = False) -> str:
        rectify_str = ""
        if rectify:
            rectify_str = (
                "\n\nIMPORTANT: If you identify any bug/error and propose a code change, you MUST wrap the proposed fix exactly in "
                "the following XML structure so the system can apply it automatically:\n"
                "<code_fix>\n"
                "  <filepath>relative/path/to/file.py</filepath>\n"
                "  <original_code>\n"
                "// Exact block of old code to replace (must match precisely including spacing)\n"
                "  </original_code>\n"
                "  <replacement_code>\n"
                "// Exact block of new code to insert\n"
                "  </replacement_code>\n"
                "</code_fix>\n"
                "Make sure that the <original_code> block you target matches the codebase content exactly, character-for-character."
            )
        return (
            "You are a graph-aware repository QA assistant. Use the selected CodeGraph and Graphify nodes, "
            "relationships, and snippets to answer. Prefer precise relationships over broad guesses. "
            "Cite files, lines, and relevant graph nodes. If the graph context is insufficient, say so. "
            "Be extremely concise, direct, and brief in your answer. Avoid verbose explanations."
            f"{rectify_str}\n\n"
            f"Question:\n{query}\n\n"
            f"Optimized graph context:\n{context}"
        )

    def _get_total_repo_tokens(self, repo_id: str) -> int:
        from app.services.file_utils import read_text_lossy
        repo_root = self.storage.repo_source_dir(repo_id)
        if not repo_root.exists():
            return 0
        
        files = self.storage.load_files(repo_id)
        total_tokens = 0
        for repo_file in files:
            file_path = repo_root / repo_file.path
            if file_path.exists():
                try:
                    text = read_text_lossy(file_path)
                    total_tokens += self.token_service.estimate_tokens(text)
                except Exception:
                    continue
        return total_tokens


