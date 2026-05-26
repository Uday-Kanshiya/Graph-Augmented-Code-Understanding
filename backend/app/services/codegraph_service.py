from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

from app.models.schemas import GraphDocument, GraphEdge, GraphNode, RepoFile
from app.services.file_utils import read_text_lossy, source_snippet


class CodeGraphService:
    def build(self, repo_id: str, repo_root: Path, files: list[RepoFile]) -> GraphDocument:
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        symbol_index: dict[str, str] = {}
        module_index: dict[str, str] = {}
        pending_imports: list[tuple[str, str, str]] = []
        pending_calls: list[tuple[str, str, int | None]] = []
        pending_inherits: list[tuple[str, str]] = []
        warnings: list[str] = []

        for repo_file in files:
            rel_path = repo_file.path
            path = repo_root / rel_path
            text = read_text_lossy(path)
            module_name = self._module_name(rel_path)
            module_id = self._node_id("module", rel_path, module_name, 1)
            module_node = GraphNode(
                node_id=module_id,
                node_type="module",
                label=module_name,
                file_path=rel_path,
                line_start=1,
                line_end=max(1, repo_file.line_count),
                source_snippet=text[:1200],
                metadata={"path": rel_path},
            )
            nodes[module_id] = module_node
            symbol_index[module_name] = module_id
            module_index[module_name] = module_id

            try:
                tree = ast.parse(text, filename=rel_path)
            except SyntaxError as exc:
                warnings.append(f"AST parse failed for {rel_path}: {exc}")
                continue

            parent_stack: list[tuple[str, str]] = [("module", module_id)]
            for child in tree.body:
                self._walk_top_level(
                    child,
                    rel_path,
                    text,
                    module_name,
                    parent_stack,
                    nodes,
                    edges,
                    symbol_index,
                    pending_imports,
                    pending_calls,
                    pending_inherits,
                )

        for source_id, import_name, imported_module in pending_imports:
            target_id = module_index.get(imported_module) or module_index.get(import_name)
            if target_id is None:
                target_id = self._node_id("import", imported_module, import_name, 0)
                nodes.setdefault(
                    target_id,
                    GraphNode(
                        node_id=target_id,
                        node_type="import",
                        label=import_name,
                        metadata={"module": imported_module, "external": True},
                    ),
                )
            self._add_edge(edges, "imports", source_id, target_id, score=1.0)

        for source_id, call_name, line_no in pending_calls:
            target_id = symbol_index.get(call_name)
            if target_id is None:
                short = call_name.split(".")[-1]
                target_id = symbol_index.get(short)
            if target_id is None:
                target_id = self._node_id("external_symbol", call_name, call_name, line_no or 0)
                nodes.setdefault(
                    target_id,
                    GraphNode(
                        node_id=target_id,
                        node_type="external_symbol",
                        label=call_name,
                        metadata={"external": True},
                    ),
                )
            self._add_edge(edges, "calls", source_id, target_id, score=0.7)

        for class_id, base_name in pending_inherits:
            target_id = symbol_index.get(base_name) or symbol_index.get(base_name.split(".")[-1])
            if target_id is None:
                target_id = self._node_id("external_symbol", base_name, base_name, 0)
                nodes.setdefault(
                    target_id,
                    GraphNode(
                        node_id=target_id,
                        node_type="external_symbol",
                        label=base_name,
                        metadata={"external": True},
                    ),
                )
            self._add_edge(edges, "inherits", class_id, target_id, score=0.8)

        return GraphDocument(
            repo_id=repo_id,
            source="codegraph",
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            warnings=warnings,
        )

    def _walk_top_level(
        self,
        node: ast.AST,
        rel_path: str,
        text: str,
        module_name: str,
        parent_stack: list[tuple[str, str]],
        nodes: dict[str, GraphNode],
        edges: dict[str, GraphEdge],
        symbol_index: dict[str, str],
        pending_imports: list[tuple[str, str, str]],
        pending_calls: list[tuple[str, str, int | None]],
        pending_inherits: list[tuple[str, str]],
    ) -> None:
        parent_kind, parent_id = parent_stack[-1]

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for import_name, imported_module in self._import_names(node):
                pending_imports.append((parent_id, import_name, imported_module))
            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = self._qualname(module_name, parent_stack, nodes, node.name)
            node_type = "method" if parent_kind == "class" else "function"
            graph_node = self._symbol_node(node_type, rel_path, text, node.name, qualname, node)
            nodes[graph_node.node_id] = graph_node
            symbol_index[node.name] = graph_node.node_id
            symbol_index[qualname] = graph_node.node_id
            self._add_edge(edges, "contains", parent_id, graph_node.node_id, score=1.0)
            for call in self._calls(node):
                pending_calls.append((graph_node.node_id, call[0], call[1]))
            return

        if isinstance(node, ast.ClassDef):
            qualname = self._qualname(module_name, parent_stack, nodes, node.name)
            graph_node = self._symbol_node("class", rel_path, text, node.name, qualname, node)
            nodes[graph_node.node_id] = graph_node
            symbol_index[node.name] = graph_node.node_id
            symbol_index[qualname] = graph_node.node_id
            self._add_edge(edges, "contains", parent_id, graph_node.node_id, score=1.0)
            for base in node.bases:
                base_name = self._call_or_name(base)
                if base_name:
                    pending_inherits.append((graph_node.node_id, base_name))
            parent_stack.append(("class", graph_node.node_id))
            for child in node.body:
                self._walk_top_level(
                    child,
                    rel_path,
                    text,
                    module_name,
                    parent_stack,
                    nodes,
                    edges,
                    symbol_index,
                    pending_imports,
                    pending_calls,
                    pending_inherits,
                )
            parent_stack.pop()
            return

    def _symbol_node(
        self,
        node_type: str,
        rel_path: str,
        text: str,
        label: str,
        qualname: str,
        node: ast.AST,
    ) -> GraphNode:
        line_start = getattr(node, "lineno", 1)
        line_end = getattr(node, "end_lineno", line_start)
        return GraphNode(
            node_id=self._node_id(node_type, rel_path, qualname, line_start),
            node_type=node_type,
            label=label,
            file_path=rel_path,
            line_start=line_start,
            line_end=line_end,
            source_snippet=source_snippet(text, line_start, line_end),
            metadata={"qualified_name": qualname, "ast_type": type(node).__name__},
        )

    def _module_name(self, rel_path: str) -> str:
        path = Path(rel_path)
        parts = list(path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) or path.stem

    def _qualname(
        self,
        module_name: str,
        parent_stack: list[tuple[str, str]],
        nodes: dict[str, GraphNode],
        name: str,
    ) -> str:
        class_parts = [nodes[node_id].label for kind, node_id in parent_stack if kind == "class" and node_id in nodes]
        return ".".join([module_name, *class_parts, name])

    def _node_id(self, node_type: str, rel_path: str, label: str, line: int) -> str:
        raw = f"{node_type}:{rel_path}:{label}:{line}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return f"codegraph:{node_type}:{digest}:{label}"

    def _edge_id(self, edge_type: str, source_id: str, target_id: str) -> str:
        raw = f"{edge_type}:{source_id}:{target_id}"
        return f"codegraph:edge:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"

    def _add_edge(
        self,
        edges: dict[str, GraphEdge],
        edge_type: str,
        source_id: str,
        target_id: str,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        edge_id = self._edge_id(edge_type, source_id, target_id)
        edges.setdefault(
            edge_id,
            GraphEdge(
                edge_id=edge_id,
                edge_type=edge_type,
                source_node=source_id,
                target_node=target_id,
                score=score,
                metadata=metadata or {},
            ),
        )

    def _import_names(self, node: ast.Import | ast.ImportFrom) -> list[tuple[str, str]]:
        if isinstance(node, ast.Import):
            return [(alias.asname or alias.name, alias.name) for alias in node.names]
        module = node.module or ""
        return [(alias.asname or alias.name, f"{module}.{alias.name}".strip(".")) for alias in node.names]

    def _calls(self, function_node: ast.AST) -> list[tuple[str, int | None]]:
        calls: list[tuple[str, int | None]] = []
        for node in ast.walk(function_node):
            if isinstance(node, ast.Call):
                name = self._call_or_name(node.func)
                if name:
                    calls.append((name, getattr(node, "lineno", None)))
        return calls

    def _call_or_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._call_or_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        if isinstance(node, ast.Call):
            return self._call_or_name(node.func)
        return None
