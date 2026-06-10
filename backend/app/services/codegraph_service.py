from __future__ import annotations

import hashlib
import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from app.models.schemas import GraphDocument, GraphEdge, GraphNode, RepoFile
from app.services.file_utils import read_text_lossy


class CodeGraphService:
    def build(self, repo_id: str, repo_root: Path, files: list[RepoFile]) -> GraphDocument:
        # Generate temporary HTML file in the repository root to avoid sandbox/permission issues
        temp_html_name = f"codegraph_temp_{repo_id}_{uuid.uuid4().hex}.html"
        temp_html_path = repo_root / temp_html_name
        
        try:
            # Call the CLI tool directly. If it fails, raise RuntimeError immediately.
            cmd = ["codegraph", "--output", str(temp_html_path), str(repo_root)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                raise RuntimeError(
                    f"CodeGraph CLI execution failed with exit code {result.returncode}.\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}"
                )
            
            if not temp_html_path.exists():
                raise RuntimeError(
                    f"CodeGraph CLI succeeded but did not produce expected output at {temp_html_path}.\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}"
                )
                
            # Read and parse the generated HTML file
            html_content = temp_html_path.read_text(encoding="utf-8")
            graph_data = self._parse_codegraph_html(html_content)
            
        finally:
            # Clean up the temporary file immediately
            if temp_html_path.exists():
                try:
                    temp_html_path.unlink()
                except Exception:
                    pass

        raw_nodes = graph_data.get("nodes", [])
        raw_links = graph_data.get("links", [])
        
        # Build maps for parsing
        module_path_map = {}
        for node in raw_nodes:
            if node.get("type") == "module" and "fullPath" in node:
                module_path_map[node["id"]] = node["fullPath"]
                
        # Read file contents lazily to extract snippets
        file_contents_cache: dict[str, str] = {}
        def get_file_content(rel_path: str) -> str:
            if rel_path not in file_contents_cache:
                path = repo_root / rel_path
                if path.exists() and path.is_file():
                    file_contents_cache[rel_path] = read_text_lossy(path)
                else:
                    file_contents_cache[rel_path] = ""
            return file_contents_cache[rel_path]

        nodes: list[GraphNode] = []
        for node in raw_nodes:
            node_id = node.get("id")
            if not node_id:
                continue
                
            node_type = node.get("type", "entity")
            label = node.get("label", node_id)
            lines_count = node.get("lines", 0)
            
            file_path: str | None = None
            line_start: int | None = None
            line_end: int | None = None
            source_snippet: str | None = None
            metadata: dict[str, Any] = {}
            
            if node_type == "module":
                file_path = node.get("fullPath")
                line_start = 1
                line_end = max(1, lines_count)
                metadata["path"] = file_path
                if file_path:
                    content = get_file_content(file_path)
                    if content:
                        source_snippet = content[:800]
                        
            elif node_type == "entity":
                parent_id = node.get("parent")
                file_path = module_path_map.get(parent_id) if parent_id else None
                entity_type = node.get("entityType", "function")
                metadata["entity_type"] = entity_type
                metadata["parent"] = parent_id
                
                if file_path:
                    content = get_file_content(file_path)
                    if content:
                        loc = self._locate_symbol(content, label, entity_type)
                        if loc:
                            line_start, file_len = loc
                            line_end = line_start + max(0, lines_count - 1)
                            # Extract lines for snippet
                            lines = content.splitlines()
                            start_idx = max(0, line_start - 1)
                            end_idx = min(len(lines), start_idx + (lines_count if lines_count > 0 else 20))
                            segment = "\n".join(lines[start_idx:end_idx])
                            if len(segment) > 800:
                                segment = segment[:800] + "\n..."
                            source_snippet = segment
                        else:
                            line_start = 1
                            line_end = 1
                            
            elif node_type == "external":
                metadata["external"] = True
                
            nodes.append(
                GraphNode(
                    node_id=node_id,
                    node_type=node_type,
                    label=label,
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    source_snippet=source_snippet,
                    metadata=metadata,
                )
            )

        edges: list[GraphEdge] = []
        for link in raw_links:
            source = link.get("source")
            target = link.get("target")
            edge_type = link.get("type", "dependency")
            if not source or not target:
                continue
                
            edge_id = self._edge_id(edge_type, source, target)
            score = 0.7
            if edge_type == "module-entity":
                score = 1.0
            elif edge_type == "module-module":
                score = 0.9
                
            edges.append(
                GraphEdge(
                    edge_id=edge_id,
                    edge_type=edge_type,
                    source_node=source,
                    target_node=target,
                    score=score,
                    metadata={},
                )
            )

        return GraphDocument(
            repo_id=repo_id,
            source="codegraph",
            nodes=nodes,
            edges=edges,
            warnings=[],
        )

    def query(self, graph: GraphDocument, query: str, max_result_nodes: int = 8) -> tuple[list[GraphNode], list[GraphEdge]]:
        """
        Execute a structural query against the CodeGraph.
        
        Returns nodes matching the query and their neighborhood edges.
        For CodeGraph, we use 2 anchor nodes since CodeGraph has more detailed nodes.
        """
        if not graph or not graph.nodes:
            return [], []
        
        # Extract query terms (words > 1 char)
        query_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(term) > 1]
        
        # Score nodes based on query term matches in label, file_path, and source_snippet
        def score_node(node: GraphNode) -> float:
            score = 0.0
            label = (node.label or "").lower()
            path = (node.file_path or "").lower()
            snippet = (node.source_snippet or "").lower()
            metadata = " ".join(str(v).lower() for v in (node.metadata or {}).values())
            
            for term in query_terms:
                if term in label:
                    score += 10
                if term in path:
                    score += 5
                if term in snippet:
                    score += 4
                if term in metadata:
                    score += 2
            return score
        
        # Score all nodes
        scored_nodes = [(score_node(node), node) for node in graph.nodes]
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        # If no matches found, use node degree (connectivity) as fallback
        if not scored_nodes or scored_nodes[0][0] == 0:
            def node_degree(node_id: str) -> int:
                return sum(1 for edge in graph.edges if edge.source_node == node_id or edge.target_node == node_id)
            
            scored_nodes = [(node_degree(node.node_id), node) for node in graph.nodes]
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        # Select 2 anchor nodes (CodeGraph has more nodes)
        anchor_count = min(2, len(scored_nodes))
        selected_nodes = [node for _, node in scored_nodes[:anchor_count]]
        anchor_ids = {node.node_id for node in selected_nodes}
        
        # Expand to neighbors via edges
        neighbor_ids = set()
        for edge in graph.edges:
            if edge.source_node in anchor_ids and edge.target_node not in anchor_ids:
                neighbor_ids.add(edge.target_node)
            if edge.target_node in anchor_ids and edge.source_node not in anchor_ids:
                neighbor_ids.add(edge.source_node)
        
        # Add neighbor nodes to selection up to max_result_nodes
        node_by_id = {node.node_id: node for node in graph.nodes}
        neighbor_nodes = [node_by_id[nid] for nid in neighbor_ids if nid in node_by_id]
        
        selected_nodes.extend(neighbor_nodes[:max(0, max_result_nodes - len(selected_nodes))])
        
        # Fill remaining slots with high-scored nodes if needed
        if len(selected_nodes) < max_result_nodes:
            additional_ids = {node.node_id for node in selected_nodes}
            additional = [node for _, node in scored_nodes if node.node_id not in additional_ids]
            selected_nodes.extend(additional[:max_result_nodes - len(selected_nodes)])
        
        selected_nodes = selected_nodes[:max_result_nodes]
        
        # Build edges connecting selected nodes
        selected_node_ids = {node.node_id for node in selected_nodes}
        selected_edges = [edge for edge in graph.edges if edge.source_node in selected_node_ids and edge.target_node in selected_node_ids]
        
        return selected_nodes, selected_edges

    def _parse_codegraph_html(self, html_content: str) -> dict:
        start_marker = "const graphData = "
        idx = html_content.find(start_marker)
        if idx == -1:
            raise ValueError("Could not find const graphData in codegraph HTML")
        
        start_pos = idx + len(start_marker)
        end_pos = html_content.find("};", start_pos)
        if end_pos == -1:
            raise ValueError("Could not find ending }; for graphData in codegraph HTML")
        
        json_str = html_content[start_pos:end_pos + 1].strip()
        return json.loads(json_str)

    def _locate_symbol(self, file_content: str, label: str, entity_type: str) -> tuple[int, int] | None:
        lines = file_content.splitlines()
        
        # Build patterns based on entity type
        patterns = []
        if entity_type == "class":
            patterns = [
                re.compile(r'\bclass\s+' + re.escape(label) + r'\b'),
                re.compile(r'\bstruct\s+' + re.escape(label) + r'\b'),
                re.compile(r'\binterface\s+' + re.escape(label) + r'\b'),
                re.compile(r'\btype\s+' + re.escape(label) + r'\b'),
            ]
        else: # function or method
            patterns = [
                re.compile(r'\bdef\s+' + re.escape(label) + r'\b'),
                re.compile(r'\bfunction\s+' + re.escape(label) + r'\b'),
                re.compile(r'\bfunc\s+(?:\([^)]*\)\s+)?' + re.escape(label) + r'\b'),
                re.compile(r'\b' + re.escape(label) + r'\s*=\s*(?:async\s*)?\([^)]*\)\s*=>'),
                re.compile(r'\b' + re.escape(label) + r'\s*\([^)]*\)\s*\{'),
            ]
            
        fallback_pattern = re.compile(r'\b' + re.escape(label) + r'\b')
        
        for pattern in patterns:
            for idx, line in enumerate(lines):
                if pattern.search(line):
                    return idx + 1, len(lines)
                    
        for idx, line in enumerate(lines):
            if fallback_pattern.search(line):
                return idx + 1, len(lines)
                
        return None

    def _edge_id(self, edge_type: str, source_id: str, target_id: str) -> str:
        raw = f"{edge_type}:{source_id}:{target_id}"
        return f"codegraph:edge:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"
