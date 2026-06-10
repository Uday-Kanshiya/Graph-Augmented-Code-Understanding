from __future__ import annotations

import time
from uuid import uuid4

from app.models.schemas import CountType, QueryRecord, TokenMeasurement
from app.services.graph_retrieval_service import GraphRetrievalService
from app.services.llm.base import LLMConfigurationError, LLMProvider
from app.services.retrieval_service import RetrievalService
from app.services.storage import LocalStorage
from app.services.token_service import TokenService


class ChatService:
    def __init__(
        self,
        storage: LocalStorage,
        graph_retrieval_service: GraphRetrievalService,
        token_service: TokenService,
        llm_provider: LLMProvider,
        retrieval_service: RetrievalService | None = None,
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

    def graph_error_analysis(self, repo_id: str, query: str, session_id: str | None = None, source_selection: str = "merged", max_nodes: int = 8) -> QueryRecord:
        if self.storage.load_repo_metadata(repo_id) is None:
            raise ValueError("Repo not found.")
        session_id = session_id or uuid4().hex
        query_id = uuid4().hex
        started = time.perf_counter()

        graphify = self.storage.load_graphify(repo_id)
        codegraph = self.storage.load_codegraph(repo_id)
        if graphify and graphify.nodes:
            selected_source = "graphify"
        elif codegraph and codegraph.nodes:
            selected_source = "codegraph"
        else:
            selected_source = source_selection

        graph_context = self.graph_retrieval_service.build_context(
            repo_id,
            query,
            max_nodes=max_nodes,
            source_selection=selected_source,
        )
        prompt = self._graph_error_prompt(query, graph_context.context)
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
        self.storage.append_log(repo_id, "chat-graph-error", "info", f"Error analysis query {query_id} finished with status {status}.")
        return record

    def _graph_error_prompt(self, query: str, context: str) -> str:
        return (
            "You are a graph-aware repository QA assistant. Use only the selected graph context to diagnose problems. "
            "Do not use the full codebase or any code outside the graph. "
            "First, determine whether the reported error or bug appears to be present in the code implied by the graph. "
            "If it is present, explain why and suggest concrete fixes. "
            "If it cannot be confirmed from the graph alone, say so clearly. "
            "Be concise, direct, and focused on the graph evidence."
            f"\n\nQuestion:\n{query}\n\nGraph context:\n{context}"
        )

    def _prompt_measurement(self, prompt: str) -> TokenMeasurement:
        try:
            return self.llm_provider.count_tokens(prompt, "llm_prompt_tokens")
        except LLMConfigurationError as exc:
            return self.token_service.measure_estimated("llm_prompt_tokens", prompt, notes=str(exc))

    def get_rectify_instructions(self) -> str:
        return (
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

    def _standard_prompt(self, query: str, context: str, rectify: bool = False) -> str:
        rectify_str = self.get_rectify_instructions() if rectify else ""
        return (
            "You are a repository QA assistant. Answer only from the provided source context. "
            "Cite file paths and line ranges when possible. If the context is insufficient, say so. "
            "Be extremely concise, direct, and brief in your answer. Avoid verbose explanations."
            f"{rectify_str}\n\n"
            f"Question:\n{query}\n\n"
            f"Source context:\n{context}"
        )

    def _graph_prompt(self, query: str, context: str, rectify: bool = False) -> str:
        rectify_str = self.get_rectify_instructions() if rectify else ""
            
        import re
        is_codegraph = "codegraph" in context.lower()
        if is_codegraph:
            cleaned_context = re.sub(r"Graphify query:[^\n]*\n?", "", context, flags=re.IGNORECASE)
            cleaned_context = re.sub(r"Selected Graph Nodes:\n?", "", cleaned_context, flags=re.IGNORECASE)
            cleaned_context = cleaned_context.strip()
            
            return (
                "You are a graph-aware repository QA assistant. Use the selected CodeGraph nodes and relationships to answer the question. "
                "Prefer precise relationships over broad guesses. "
                "Cite files, lines, and relevant symbols. If the graph context is insufficient, say so. "
                "Be extremely concise, direct, and brief in your answer. Avoid verbose explanations."
                f"{rectify_str}\n\n"
                f"Question:\n{query}\n\n"
                f"{cleaned_context}"
            )
            
        return (
            "You are a graph-aware repository QA assistant. Use the selected Graphify nodes and relationships first, "
            "then fall back to CodeGraph details if needed. Prefer precise relationships over broad guesses. "
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


