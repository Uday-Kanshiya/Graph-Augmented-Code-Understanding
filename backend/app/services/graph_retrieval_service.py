from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.schemas import GraphDocument, GraphEdge, GraphNode, SourceSnippet, TokenMeasurement
from app.services.file_utils import read_text_lossy
import subprocess
import json
import tempfile
from app.services.storage import LocalStorage
from app.services.token_service import TokenService
from app.services.graphify_service import GraphifyService
from app.services.codegraph_service import CodeGraphService


@dataclass
class GraphRetrievalResult:
    context: str
    snippets: list[SourceSnippet]
    selected_nodes: list[GraphNode]
    selected_edges: list[GraphEdge]
    token_measurement: TokenMeasurement


class GraphRetrievalService:
    """Query-driven graph retrieval using Graphify and CodeGraph query engines.

    This implementation uses GraphifyService.query() and CodeGraphService.query()
    to execute semantic/structural queries against their respective graphs,
    then passes the results directly to the LLM for explanation.
    """

    def __init__(self, storage: LocalStorage, token_service: TokenService) -> None:
        self.storage = storage
        self.token_service = token_service
        self.graphify_service = GraphifyService(storage)
        self.codegraph_service = CodeGraphService()

    def build_context(self, repo_id: str, query: str, max_nodes: int = 8, source_selection: str = "merged") -> GraphRetrievalResult:
        graphify = self.storage.load_graphify(repo_id)
        codegraph = self.storage.load_codegraph(repo_id)

        if source_selection == "graphify":
            graph = graphify
        elif source_selection == "codegraph":
            graph = codegraph
        else:
            graph = graphify or codegraph

        if graph is None or not graph.nodes:
            if source_selection == "graphify":
                raise ValueError("Graphify output is not available for this repository.")
            if source_selection == "codegraph":
                raise ValueError("CodeGraph output is not available for this repository.")
            return self._fallback_context(repo_id)

        # Use the appropriate query service based on graph source
        if source_selection == "merged" and graphify and codegraph:
            graphify_result = self._build_context_for_graph(repo_id, query, max_nodes, graphify)
            codegraph_result = self._build_context_for_graph(repo_id, query, max_nodes, codegraph)
            merged_context = "\n\n---\n\n".join([graphify_result.context, codegraph_result.context])
            merged_snippets = graphify_result.snippets + codegraph_result.snippets
            merged_nodes = graphify_result.selected_nodes + codegraph_result.selected_nodes
            merged_edges = graphify_result.selected_edges + codegraph_result.selected_edges
            measurement = self.token_service.measure_estimated("codegraph_graphify_optimized_context", merged_context)
            return GraphRetrievalResult(
                context=merged_context,
                snippets=merged_snippets,
                selected_nodes=merged_nodes,
                selected_edges=merged_edges,
                token_measurement=measurement,
            )

        if source_selection == "merged":
            graph = graphify or codegraph

        graph_context = self._build_context_for_graph(repo_id, query, max_nodes, graph)
        return graph_context

    def _build_context_for_graph(self, repo_id: str, query: str, max_nodes: int, graph: GraphDocument) -> GraphRetrievalResult:
        if graph.source == "graphify":
            selected_nodes, selected_edges = [], []
            graphify_cli_available = False
            if graph.raw_output_path:
                raw_path = Path(graph.raw_output_path)
                if not raw_path.is_absolute():
                    raw_path = self.storage.repo_source_dir(repo_id) / raw_path

                if raw_path.exists():
                    if raw_path.is_file() and raw_path.name == "graph.json":
                        graphify_cli_available = True
                    elif raw_path.is_dir() and (raw_path / "graph.json").exists():
                        graphify_cli_available = True
                    elif (raw_path / "graph.json").exists():
                        graphify_cli_available = True

            if graphify_cli_available:
                try:
                    repo_root = self.storage.repo_source_dir(repo_id)
                    proc = subprocess.run(
                        ["graphify", "query", query],
                        cwd=str(repo_root),
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    cli_output = proc.stdout.strip() or proc.stderr.strip()
                    if cli_output:
                        context_text = f"Graphify CLI output for query: {query}\n\n{cli_output}"
                        measurement = self.token_service.measure_estimated(
                            "codegraph_graphify_optimized_context", context_text
                        )
                        return GraphRetrievalResult(
                            context=context_text,
                            snippets=[SourceSnippet(file_path="graphify:cli", line_start=0, line_end=0, text=cli_output, source="graphify_cli")],
                            selected_nodes=[],
                            selected_edges=[],
                            token_measurement=measurement,
                        )
                except Exception:
                    pass

            selected_nodes, selected_edges = self.graphify_service.query(graph, query, max_nodes)
        else:
            selected_nodes, selected_edges = [], []
            try:
                repo_root = self.storage.repo_source_dir(repo_id)
                script = r"""
                (async () => {
                  try {
                    const { CodeGraph } = require('@colbymchenry/codegraph');
                    const cg = await CodeGraph.open(process.cwd());
                    if (cg.indexAll) await cg.indexAll();
                    const query = process.argv[2] || '';
                    const ctx = await cg.buildContext(query, { maxNodes: %d, includeCode: true, format: 'markdown' });
                    console.log(JSON.stringify({ context: ctx }));
                    if (cg.close) await cg.close();
                  } catch (e) {
                    console.error(e && e.stack ? e.stack : e);
                    process.exit(2);
                  }
                })();
                """ % (max_nodes)

                script_path = repo_root / ".codegraph_query.js"
                script_path.write_text(script, encoding="utf-8")
                proc = subprocess.run(
                    ["node", str(script_path), query],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                stdout = proc.stdout.strip()
                if stdout:
                    try:
                        parsed = json.loads(stdout)
                        cli_context = parsed.get("context") or ""
                        context_text = f"CodeGraph CLI output for query: {query}\n\n{cli_context}"
                        measurement = self.token_service.measure_estimated(
                            "codegraph_graphify_optimized_context", context_text
                        )
                        try:
                            script_path.unlink()
                        except Exception:
                            pass
                        return GraphRetrievalResult(
                            context=context_text,
                            snippets=[SourceSnippet(file_path="codegraph:cli", line_start=0, line_end=0, text=cli_context, source="codegraph_cli")],
                            selected_nodes=[],
                            selected_edges=[],
                            token_measurement=measurement,
                        )
                    except json.JSONDecodeError:
                        cli_output = stdout
                        context_text = f"CodeGraph CLI output for query: {query}\n\n{cli_output}"
                        measurement = self.token_service.measure_estimated(
                            "codegraph_graphify_optimized_context", context_text
                        )
                        try:
                            script_path.unlink()
                        except Exception:
                            pass
                        return GraphRetrievalResult(
                            context=context_text,
                            snippets=[SourceSnippet(file_path="codegraph:cli", line_start=0, line_end=0, text=cli_output, source="codegraph_cli")],
                            selected_nodes=[],
                            selected_edges=[],
                            token_measurement=measurement,
                        )
            except Exception:
                selected_nodes, selected_edges = self.codegraph_service.query(graph, query, max_nodes)

        snippets = [self._node_to_snippet(node) for node in selected_nodes if node.source_snippet]
        context_text = self._build_graph_context_text(query, selected_nodes, selected_edges)
        measurement = self.token_service.measure_estimated("codegraph_graphify_optimized_context", context_text)

        return GraphRetrievalResult(
            context=context_text,
            snippets=snippets,
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
            token_measurement=measurement,
        )

    def _fallback_context(self, repo_id: str) -> GraphRetrievalResult:
        source_dir = self.storage.repo_source_dir(repo_id)
        snippets: list[SourceSnippet] = []
        context_text = ""

        py_files = list(source_dir.rglob("*.py"))
        if py_files:
            chosen = py_files[0]
            text = read_text_lossy(chosen)[:8000]
            context_text = f"Fallback code excerpt from {chosen.name}:\n{text}"
            snippets.append(SourceSnippet(file_path=str(chosen.relative_to(source_dir)), line_start=1, line_end=min(200, text.count('\n') + 1), text=text, source="fallback"))

        measurement = self.token_service.measure_estimated("codegraph_graphify_optimized_context", context_text)
        return GraphRetrievalResult(context=context_text, snippets=snippets, selected_nodes=[], selected_edges=[], token_measurement=measurement)

    def _node_to_snippet(self, node: GraphNode) -> SourceSnippet:
        return SourceSnippet(
            file_path=node.file_path or "unknown",
            line_start=node.line_start or 0,
            line_end=node.line_end or 0,
            text=node.source_snippet or "",
            source="graphify_node",
        )

    def _build_graph_context_text(self, query: str, nodes: list[GraphNode], edges: list[GraphEdge]) -> str:
        lines = [f"Graphify query: {query}", "Selected Graph Nodes:"]
        for node in nodes:
            lines.append("---")
            lines.append(f"Node ID: {node.node_id}")
            lines.append(f"Type: {node.node_type}")
            if node.label:
                lines.append(f"Label: {node.label}")
            if node.file_path:
                lines.append(f"File: {node.file_path}:{node.line_start}-{node.line_end}")
            snippet = node.source_snippet or node.metadata.get("source_snippet") or ""
            if snippet:
                lines.append("Source snippet:")
                lines.append(snippet)
        if edges:
            lines.append("\nSelected Graph Edges:")
            for edge in edges:
                lines.append(f"- {edge.source_node} --{edge.edge_type}--> {edge.target_node}")
        return "\n".join(lines)
