from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import GraphEdge, GraphNode, SourceSnippet, TokenMeasurement
from app.services.storage import LocalStorage
from app.services.token_service import TokenService


@dataclass
class GraphRetrievalResult:
    context: str
    snippets: list[SourceSnippet]
    selected_nodes: list[GraphNode]
    selected_edges: list[GraphEdge]
    token_measurement: TokenMeasurement


class GraphRetrievalService:
    def __init__(self, storage: LocalStorage, token_service: TokenService) -> None:
        self.storage = storage
        self.token_service = token_service

    def _compute_pagerank(self, nodes: list[GraphNode], edges: list[GraphEdge], damping: float = 0.85, max_iter: int = 20) -> dict[str, float]:
        n = len(nodes)
        if n == 0:
            return {}
        
        # Initialize PageRank equally
        pr = {node.node_id: 1.0 / n for node in nodes}
        
        # Build adjacency and incoming mappings
        out_degree: dict[str, int] = {}
        incoming: dict[str, list[str]] = {node.node_id: [] for node in nodes}
        
        for edge in edges:
            src, tgt = edge.source_node, edge.target_node
            if src in pr and tgt in pr:
                incoming[tgt].append(src)
                out_degree[src] = out_degree.get(src, 0) + 1
                
        # Power iteration
        for _ in range(max_iter):
            new_pr = {}
            # Redistribute sink rank (0 out-degree) equally
            sink_sum = sum(pr[node_id] for node_id, deg in out_degree.items() if deg == 0)
            
            for node in nodes:
                nid = node.node_id
                rank = (1.0 - damping) / n
                rank += damping * (sink_sum / n)
                
                for src in incoming[nid]:
                    rank += damping * (pr[src] / out_degree[src])
                    
                new_pr[nid] = rank
            pr = new_pr
            
        return pr

    def _extract_node_signature(self, node: GraphNode) -> str:
        # Signature is lightweight: node name, type, and the first few lines of its code (if present)
        parts = [node.label or "", node.node_type or ""]
        if node.source_snippet:
            lines = node.source_snippet.splitlines()[:5]
            parts.extend(lines)
        return "\n".join(parts)

    def _format_neighbor_snippet(self, text: str, limit_chars: int) -> str:
        if not text:
            return ""
        if len(text) <= limit_chars:
            return text
        
        prefix = text[:limit_chars]
        remaining = text[limit_chars:]
        extra_lines = []
        
        for line in remaining.splitlines():
            line_strip = line.strip()
            # Catch function/method definitions and class headers
            is_declaration = (
                line_strip.startswith("def ") or 
                line_strip.startswith("class ") or 
                line_strip.startswith("function ") or
                line_strip.startswith("async ") or 
                "export " in line_strip or 
                "interface " in line_strip or
                line_strip.startswith("public ") or
                line_strip.startswith("private ")
            )
            # Catch docstring headers and comment structures
            is_docstring = (
                line_strip.startswith('"""') or 
                line_strip.startswith("'''") or 
                line_strip.startswith("//") or 
                line_strip.startswith("/*") or 
                line_strip.startswith("*") or
                line_strip.startswith("#")
            )
            if is_declaration or is_docstring:
                extra_lines.append(line)
                
        if extra_lines:
            return prefix + "\n\n... [Snippet Truncated - Declarations & Headers Expanded] ...\n" + "\n".join(extra_lines)
        return prefix + "\n\n... [Snippet Truncated] ..."

    def build_context(self, repo_id: str, query: str, max_nodes: int = 8, source_selection: str = "merged") -> GraphRetrievalResult:
        codegraph = self.storage.load_codegraph(repo_id)
        graphify = self.storage.load_graphify(repo_id)
        if codegraph is None:
            raise ValueError("CodeGraph output not found for repo.")

        # Source Selection and Dynamic Graph Merging
        if source_selection == "codegraph" or not graphify:
            all_nodes = codegraph.nodes
            all_edges = codegraph.edges
        elif source_selection == "graphify":
            all_nodes = graphify.nodes
            all_edges = graphify.edges
        else: # "merged"
            cg_module_map = {n.file_path: n.node_id for n in codegraph.nodes if n.node_type == "module" and n.file_path}
            
            merged_nodes = list(codegraph.nodes)
            node_id_rewrites = {}
            
            for n in graphify.nodes:
                file_path = n.file_path or n.metadata.get("file_path") or n.metadata.get("path")
                if file_path and file_path in cg_module_map:
                    target_id = cg_module_map[file_path]
                    node_id_rewrites[n.node_id] = target_id
                    for idx, node in enumerate(merged_nodes):
                        if node.node_id == target_id:
                            merged_nodes[idx] = node.model_copy(
                                update={"metadata": {**node.metadata, **n.metadata, "merged_from_graphify": True}}
                            )
                            break
                else:
                    merged_nodes.append(n)
                    
            merged_edges = []
            seen_edges = set()
            for edge in (codegraph.edges + graphify.edges):
                src = node_id_rewrites.get(edge.source_node, edge.source_node)
                tgt = node_id_rewrites.get(edge.target_node, edge.target_node)
                edge_key = (src, edge.edge_type, tgt)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    merged_edges.append(edge.model_copy(update={"source_node": src, "target_node": tgt}))
                
            all_nodes = merged_nodes
            all_edges = merged_edges

        # 1. Map max_nodes limits dynamically to primary anchor and neighbor budget settings
        if max_nodes <= 8:
            max_anchors = 2
            max_neighbors = 4
            limit_chars = 500
        elif max_nodes <= 14:
            max_anchors = 4
            max_neighbors = 8
            limit_chars = 1000
        else:
            max_anchors = 8
            max_neighbors = 16
            limit_chars = 1500

        # 2. Run global PageRank centrality scoring
        pr_map = self._compute_pagerank(all_nodes, all_edges)
        
        # Sort nodes by PageRank to find the ranks and apply rank-based boosts (top 5 get 1.0 down to 0.2)
        sorted_by_pr = sorted(all_nodes, key=lambda n: pr_map.get(n.node_id, 0.0), reverse=True)
        pr_boost = {}
        for index, node in enumerate(sorted_by_pr):
            pr_boost[node.node_id] = max(0.0, 1.0 - index * 0.2)

        # 3. Perform dynamic Light-to-Full checks on all nodes to calculate scoring
        terms = self._terms(query)
        
        # Traceback Line Number Scorer: extract line numbers (e.g. 'line 25' or 'file.py:25' or 'L25')
        line_numbers = []
        for match in re.finditer(r"\bline\s+(\d+)\b|:(\d+)\b|\bL(\d+)\b", query, re.IGNORECASE):
            num = match.group(1) or match.group(2) or match.group(3)
            if num:
                try:
                    line_numbers.append(int(num))
                except ValueError:
                    pass

        node_scores = {}
        
        for node in all_nodes:
            # Extract lightweight signature (name, declarations, docstrings)
            sig = self._extract_node_signature(node)
            sig_lower = sig.lower()
            
            # Check if signature matches any query terms
            has_sig_match = False
            for term in terms:
                if term in sig_lower:
                    has_sig_match = True
                    break
            
            # Dynamically promote to Full Check if matching, otherwise perform Light Check
            if has_sig_match and node.source_snippet:
                haystack = " ".join(
                    filter(
                        None,
                        [
                            node.label,
                            node.node_type,
                            node.file_path or "",
                            node.source_snippet, # Scan entire snippet body
                            " ".join(str(value) for value in node.metadata.values()),
                        ],
                    )
                ).lower()
            else:
                haystack = " ".join(
                    filter(
                        None,
                        [
                            node.label,
                            node.node_type,
                            node.file_path or "",
                            sig, # Scan signature block only
                            " ".join(str(value) for value in node.metadata.values()),
                        ],
                    )
                ).lower()
                
            # Lexical scoring
            score = 0.0
            for term in terms:
                if term in node.label.lower():
                    score += 8
                if term in haystack:
                    score += haystack.count(term)
            if node.node_type in {"function", "class", "method"}:
                score += 0.2
                
            # Line boundary matching: massive boost if a mentioned traceback line falls within this node's range
            line_boost = 0.0
            if node.file_path and node.line_start is not None and node.line_end is not None:
                for line_num in line_numbers:
                    if node.line_start <= line_num <= node.line_end:
                        line_boost += 15.0
                
            # Add rank-based PageRank boost score and line boost to lexical score
            node_scores[node.node_id] = score + pr_boost.get(node.node_id, 0.0) + line_boost

        # 4. Select top K Primary Anchors based on final scores
        sorted_nodes = sorted(all_nodes, key=lambda n: node_scores.get(n.node_id, 0.0), reverse=True)
        relevant_nodes = [node for node in sorted_nodes if node_scores.get(node.node_id, 0.0) > 0.0]
        
        if relevant_nodes:
            anchors = relevant_nodes[:max_anchors]
        else:
            anchors = sorted_nodes[:max_anchors]
            
        selected_anchors = {node.node_id: node for node in anchors}

        # 5. Select top N Neighbor Nodes linked in the graph, ranked by score
        adjacent_edges: list[GraphEdge] = []
        neighbor_ids = set()
        for edge in all_edges:
            if edge.source_node in selected_anchors or edge.target_node in selected_anchors:
                adjacent_edges.append(edge)
                other_id = edge.target_node if edge.source_node in selected_anchors else edge.source_node
                if other_id not in selected_anchors:
                    neighbor_ids.add(other_id)
                    
        node_by_id = {node.node_id: node for node in all_nodes}
        neighbor_nodes = [node_by_id[nid] for nid in neighbor_ids if nid in node_by_id]
        
        # Rank the neighbors by score and keep the top N
        ranked_neighbors = sorted(neighbor_nodes, key=lambda n: node_scores.get(n.node_id, 0.0), reverse=True)
        selected_neighbors = {node.node_id: node for node in ranked_neighbors[:max_neighbors]}

        # Combine nodes and select subset of edges
        selected_nodes = list(selected_anchors.values()) + list(selected_neighbors.values())
        selected_ids = {node.node_id for node in selected_nodes}
        selected_edges = [
            edge for edge in adjacent_edges if edge.source_node in selected_ids and edge.target_node in selected_ids
        ]

        # 6. Extract snippets: Full snippets for primary anchors, smart truncated snippets for neighbors
        snippets: list[SourceSnippet] = []
        seen_snippets = set()
        
        # Anchors (Full context)
        for node in selected_anchors.values():
            if not node.file_path or not node.source_snippet:
                continue
            key = (node.file_path, node.line_start, node.line_end)
            if key not in seen_snippets:
                seen_snippets.add(key)
                snippets.append(
                    SourceSnippet(
                        file_path=node.file_path,
                        line_start=node.line_start or 1,
                        line_end=node.line_end or node.line_start or 1,
                        text=node.source_snippet,
                        source="graph_anchor",
                    )
                )
                
        # Neighbors (Smart truncated signature context)
        for node in selected_neighbors.values():
            if not node.file_path or not node.source_snippet:
                continue
            key = (node.file_path, node.line_start, node.line_end)
            if key not in seen_snippets:
                seen_snippets.add(key)
                
                # Format snippet with dynamic budget limit + signature append
                truncated_text = self._format_neighbor_snippet(node.source_snippet, limit_chars)
                snippets.append(
                    SourceSnippet(
                        file_path=node.file_path,
                        line_start=node.line_start or 1,
                        line_end=node.line_end or node.line_start or 1,
                        text=truncated_text,
                        source="graph_neighbor",
                    )
                )

        context = self._format_context(selected_nodes, selected_edges, snippets)
        measurement = self.token_service.measure_estimated("codegraph_graphify_optimized_context", context)
        
        return GraphRetrievalResult(
            context=context,
            snippets=snippets,
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
            token_measurement=measurement,
        )

    def _terms(self, query: str) -> list[str]:
        return [term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", query)]

    def _snippets(self, nodes: list[GraphNode]) -> list[SourceSnippet]:
        # Keeps compatibility with old calls if any, though build_context now generates snippets directly
        snippets: list[SourceSnippet] = []
        seen: set[tuple[str, int | None, int | None]] = set()
        for node in nodes:
            if not node.file_path or not node.source_snippet:
                continue
            key = (node.file_path, node.line_start, node.line_end)
            if key in seen:
                continue
            seen.add(key)
            snippets.append(
                SourceSnippet(
                    file_path=node.file_path,
                    line_start=node.line_start or 1,
                    line_end=node.line_end or node.line_start or 1,
                    text=node.source_snippet,
                    source="graph",
                )
            )
        return snippets

    def _format_context(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        snippets: list[SourceSnippet],
    ) -> str:
        node_lines = [
            f"- {node.node_id} [{node.node_type}] {node.label} ({node.file_path or 'external'}:{node.line_start or '-'})"
            for node in nodes
        ]
        edge_lines = [
            f"- {edge.source_node} --{edge.edge_type}--> {edge.target_node}"
            for edge in edges
        ]
        snippet_lines = [
            f"### {snippet.file_path}:{snippet.line_start}-{snippet.line_end} ({snippet.source})\n{snippet.text}"
            for snippet in snippets
        ]
        return "\n".join(
            [
                "Graph-selected nodes:",
                *node_lines,
                "",
                "Graph relationships:",
                *edge_lines,
                "",
                "Source snippets:",
                *snippet_lines,
            ]
        )

