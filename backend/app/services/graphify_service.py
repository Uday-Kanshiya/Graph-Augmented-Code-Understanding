from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from app.models.schemas import GraphDocument, GraphEdge, GraphNode
from app.services.storage import LocalStorage


class GraphifyService:
    def __init__(
        self,
        storage: LocalStorage,
        timeout_seconds: int = 120,
        cli_name: str = "graphify",
    ) -> None:
        self.storage = storage
        self.timeout_seconds = timeout_seconds
        self.cli_name = cli_name

    def run_or_fallback(self, repo_id: str, repo_root: Path, codegraph: GraphDocument) -> GraphDocument:
        cli = self._find_cli()
        raw_dir = self.storage.repo_state_dir(repo_id) / "graphify_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_log = raw_dir / "raw-output.json"

        if cli is None:
            raw_log.write_text(
                json.dumps(
                    {
                        "available": False,
                        "cli": self.cli_name,
                        "checked_python_scripts_dir": str(Path(sys.executable).resolve().parent),
                        "message": "Graphify CLI not found; using documented structural fallback.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return self._fallback(
                repo_id,
                codegraph,
                raw_log,
                [
                    "Native Graphify CLI was not found, so Stage 1 is using a transparent graphify-fallback graph derived from CodeGraph. Install optional package graphifyy only if you need native Graphify output."
                ],
            )

        try:
            work_dir = raw_dir / "run"
            work_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [cli, "update", str(repo_root), "--no-cluster"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            raw_log.write_text(
                json.dumps(
                    {
                        "available": True,
                        "cli": cli,
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            graph_json = self._find_graph_json(work_dir, repo_root)
            if result.returncode != 0:
                return self._fallback(
                    repo_id,
                    codegraph,
                    raw_log,
                    [f"Graphify CLI exited with code {result.returncode}; using structural fallback."],
                )
            if graph_json is None:
                return self._fallback(
                    repo_id,
                    codegraph,
                    raw_log,
                    ["Graphify CLI ran but graph.json was not found; using structural fallback."],
                )
            return self._normalize_graphify_json(repo_id, graph_json, raw_log)
        except Exception as exc:
            raw_log.write_text(
                json.dumps({"available": True, "cli": cli, "error": str(exc)}, indent=2),
                encoding="utf-8",
            )
            return self._fallback(repo_id, codegraph, raw_log, [f"Graphify failed: {exc}; using structural fallback."])

    def _find_cli(self) -> str | None:
        cli = shutil.which(self.cli_name)
        if cli:
            return cli
        scripts_dir = Path(sys.executable).resolve().parent
        for name in [self.cli_name, "graphify"]:
            for suffix in [".exe", ".cmd", ""]:
                candidate = scripts_dir / f"{name}{suffix}"
                if candidate.exists():
                    return str(candidate)
        return None

    def _find_graph_json(self, *roots: Path) -> Path | None:
        for root in roots:
            for candidate in root.rglob("graph.json"):
                return candidate
        return None

    def _normalize_graphify_json(self, repo_id: str, graph_json: Path, raw_log: Path) -> GraphDocument:
        data = json.loads(graph_json.read_text(encoding="utf-8", errors="replace"))
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        raw_nodes = data.get("nodes", [])
        if isinstance(raw_nodes, dict):
            raw_nodes = [{"id": key, **(value if isinstance(value, dict) else {"label": str(value)})} for key, value in raw_nodes.items()]
        for idx, node in enumerate(raw_nodes):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or node.get("node_id") or f"graphify:node:{idx}")
            metadata = {key: value for key, value in node.items() if key not in {"id", "node_id", "label", "type"}}
            nodes.append(
                GraphNode(
                    node_id=f"graphify:{node_id}",
                    node_type=str(node.get("type") or node.get("node_type") or "concept"),
                    label=str(node.get("label") or node.get("name") or node_id),
                    file_path=node.get("file_path") or node.get("path"),
                    line_start=node.get("line_start"),
                    line_end=node.get("line_end"),
                    source_snippet=node.get("source_snippet") or node.get("snippet"),
                    metadata=metadata,
                )
            )

        raw_edges = data.get("edges") or data.get("links") or []
        for idx, edge in enumerate(raw_edges):
            if not isinstance(edge, dict):
                continue
            source = edge.get("source") or edge.get("source_node") or edge.get("from")
            target = edge.get("target") or edge.get("target_node") or edge.get("to")
            if source is None or target is None:
                continue
            edge_id = str(edge.get("id") or edge.get("edge_id") or f"graphify:edge:{idx}")
            metadata = {key: value for key, value in edge.items() if key not in {"id", "edge_id", "source", "target", "source_node", "target_node", "type"}}
            edges.append(
                GraphEdge(
                    edge_id=f"graphify:{edge_id}",
                    edge_type=str(edge.get("type") or edge.get("edge_type") or "related"),
                    source_node=f"graphify:{source}",
                    target_node=f"graphify:{target}",
                    score=edge.get("score") or edge.get("weight"),
                    metadata=metadata,
                )
            )

        return GraphDocument(
            repo_id=repo_id,
            source="graphify",
            nodes=nodes,
            edges=edges,
            raw_output_path=str(raw_log),
            warnings=[],
        )

    def _fallback(self, repo_id: str, codegraph: GraphDocument, raw_log: Path, warnings: list[str]) -> GraphDocument:
        node_map: dict[str, str] = {}
        nodes: list[GraphNode] = []
        for node in codegraph.nodes:
            fallback_id = f"graphify-fallback:{node.node_id}"
            node_map[node.node_id] = fallback_id
            nodes.append(
                node.model_copy(
                    update={
                        "node_id": fallback_id,
                        "metadata": {**node.metadata, "fallback_from": "codegraph"},
                    }
                )
            )
        edges: list[GraphEdge] = []
        for edge in codegraph.edges:
            if edge.source_node not in node_map or edge.target_node not in node_map:
                continue
            edges.append(
                edge.model_copy(
                    update={
                        "edge_id": f"graphify-fallback:{edge.edge_id}",
                        "source_node": node_map[edge.source_node],
                        "target_node": node_map[edge.target_node],
                        "metadata": {**edge.metadata, "fallback_from": "codegraph"},
                    }
                )
            )
        return GraphDocument(
            repo_id=repo_id,
            source="graphify-fallback",
            nodes=nodes,
            edges=edges,
            raw_output_path=str(raw_log),
            warnings=warnings,
        )
