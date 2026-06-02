from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from app.models.schemas import GraphDocument, GraphEdge, GraphNode


class GraphifyService:
    def __init__(self, storage: Any) -> None:
        self.storage = storage

    def query(self, graph: GraphDocument, query: str, max_result_nodes: int = 8) -> tuple[list[GraphNode], list[GraphEdge]]:
        """
        Execute a semantic query against the Graphify graph.
        
        Returns nodes matching the query and their neighborhood edges.
        For Graphify, we use 3 anchor nodes since Graphify has fewer, higher-quality nodes.
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
        
        # Select 3 anchor nodes (Graphify nodes are fewer, higher-level)
        anchor_count = min(3, len(scored_nodes))
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

    def run_or_fallback(self, repo_id: str, repo_root: Path, codegraph: GraphDocument) -> GraphDocument:
        # Build Native Graphify directly from CodeGraph in Python.
        # This completely eliminates the external graphify CLI subprocess dependency.
        
        # 1. Map child nodes to parent nodes based on 'contains' edges in CodeGraph
        parent_map: dict[str, str] = {}
        for edge in codegraph.edges:
            if edge.edge_type == "contains":
                parent_map[edge.target_node] = edge.source_node

        # 2. Build a type map of CodeGraph nodes to look up their original categories
        node_type_map: dict[str, str] = {node.node_id: node.node_type for node in codegraph.nodes}

        # Helper: Find the closest macro-level ancestor (module/class/etc.) for any node
        def get_macro_ancestor(node_id: str) -> str | None:
            current = node_id
            visited = set()
            while current:
                if current in visited:
                    break
                visited.add(current)
                ntype = node_type_map.get(current)
                # Macro nodes are files, classes, components, imports, and external symbols (not functions/methods)
                if ntype and ntype not in {"function", "method"}:
                    return current
                current = parent_map.get(current)
            return None

        # 3. Filter out micro-nodes (functions/methods) and only map macro nodes to Graphify
        nodes: list[GraphNode] = []
        node_map: dict[str, str] = {} # maps CodeGraph macro node_id to Graphify node_id
        
        for node in codegraph.nodes:
            # Skip micro-level logic nodes (functions and methods) to achieve macro-level pruning
            if node.node_type in {"function", "method"}:
                continue

            graphify_id = f"graphify:{node.node_id.replace('codegraph:', '')}"
            node_map[node.node_id] = graphify_id
            
            # Map structural AST node types to design-level concepts
            node_type = "concept"
            if node.node_type == "module":
                node_type = "file"
            elif node.node_type in {"class", "struct_item"}:
                node_type = "component"
                
            nodes.append(
                GraphNode(
                    node_id=graphify_id,
                    node_type=node_type,
                    label=node.label,
                    file_path=node.file_path,
                    line_start=node.line_start,
                    line_end=node.line_end,
                    source_snippet=node.source_snippet,
                    metadata={**node.metadata, "graphify_processed": True, "native": True},
                )
            )

        # 4. Convert and lift CodeGraph structural relationships to high-level flow edges.
        # Micro-level dependencies (like functions calling other functions) are aggregated/lifted to their macro ancestors.
        unique_edges: dict[tuple[str, str, str], GraphEdge] = {}
        
        for edge in codegraph.edges:
            # Find the nearest macro ancestor of both the source and target nodes
            macro_src = get_macro_ancestor(edge.source_node)
            macro_tgt = get_macro_ancestor(edge.target_node)
            
            if not macro_src or not macro_tgt:
                continue
            
            # Skip self-loop dependencies at the macro-level (e.g. methods calling within the same class/file)
            if macro_src == macro_tgt:
                continue
                
            source_id = f"graphify:{macro_src.replace('codegraph:', '')}"
            target_id = f"graphify:{macro_tgt.replace('codegraph:', '')}"
            
            # Map structural AST edge types to dynamic execution flows
            edge_type = "flow"
            if edge.edge_type == "contains":
                edge_type = "part_of"
            elif edge.edge_type == "imports":
                edge_type = "depends_on"
            elif edge.edge_type == "calls":
                edge_type = "triggers"
            elif edge.edge_type == "inherits":
                edge_type = "extends"
                
            edge_key = (source_id, target_id, edge_type)
            graphify_edge_id = f"graphify:edge:{macro_src.replace('codegraph:', '')}:{macro_tgt.replace('codegraph:', '')}:{edge_type}"
            
            # Keep the edge with the highest score if duplicates arise during lifting
            if edge_key not in unique_edges or unique_edges[edge_key].score < edge.score:
                unique_edges[edge_key] = GraphEdge(
                    edge_id=graphify_edge_id,
                    edge_type=edge_type,
                    source_node=source_id,
                    target_node=target_id,
                    score=edge.score,
                    metadata={**edge.metadata, "graphify_processed": True, "native": True, "lifted": True},
                )
            
        return GraphDocument(
            repo_id=repo_id,
            source="graphify",
            nodes=nodes,
            edges=list(unique_edges.values()),
            raw_output_path=None,
            warnings=[],
        )
