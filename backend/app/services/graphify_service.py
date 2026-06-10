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
        # 1. Try running graphify CLI
        import subprocess
        import json
        
        json_path = repo_root / "graphify-out" / "graph.json"
        
        # Clean up old graph.json if it exists to avoid loading stale results
        if json_path.exists():
            try:
                json_path.unlink()
            except Exception:
                pass
                
        cli_success = False
        try:
            # Run graphify . inside repo_root
            proc = subprocess.run(
                ["graphify", "."],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                shell=True,
                timeout=120
            )
            if proc.returncode == 0 and json_path.exists():
                cli_success = True
        except Exception as e:
            if hasattr(self.storage, "append_log"):
                self.storage.append_log(repo_id, "graphify", "warning", f"Graphify CLI run failed: {e}")
                
        if cli_success:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                nodes: list[GraphNode] = []
                for n in data.get("nodes", []):
                    loc = n.get("source_location") or ""
                    line_start = None
                    if loc.startswith("L"):
                        try:
                            line_start = int(loc[1:])
                        except ValueError:
                            pass
                    nodes.append(
                        GraphNode(
                            node_id=n["id"],
                            node_type=n.get("file_type", "concept"),
                            label=n.get("label", n["id"]),
                            file_path=n.get("source_file"),
                            line_start=line_start,
                            line_end=None,
                            source_snippet=None,
                            metadata={k: v for k, v in n.items() if k not in {"id", "file_type", "label", "source_file", "source_location"}},
                        )
                    )
                
                edges: list[GraphEdge] = []
                for e in data.get("links", []):
                    rel = e.get("relation") or "connected_to"
                    edges.append(
                        GraphEdge(
                            edge_id=f"graphify:edge:{e['source']}:{e['target']}:{rel}",
                            edge_type=rel,
                            source_node=e["source"],
                            target_node=e["target"],
                            score=e.get("weight") or e.get("confidence_score") or 1.0,
                            metadata={k: v for k, v in e.items() if k not in {"source", "target", "relation", "weight", "confidence_score"}},
                        )
                    )
                
                if hasattr(self.storage, "append_log"):
                    self.storage.append_log(repo_id, "graphify", "info", "Successfully generated graph using Graphify CLI.")
                
                return GraphDocument(
                    repo_id=repo_id,
                    source="graphify",
                    nodes=nodes,
                    edges=edges,
                    raw_output_path=str(json_path.relative_to(repo_root)),
                    warnings=[],
                )
            except Exception as e:
                print("Unable to generate Graphify graph")
                if hasattr(self.storage, "append_log"):
                    self.storage.append_log(repo_id, "graphify", "error", f"Failed to parse graph.json: {e}")
                raise RuntimeError("Unable to generate Graphify graph") from e
        else:
            print("Unable to generate Graphify graph")
            if hasattr(self.storage, "append_log"):
                self.storage.append_log(repo_id, "graphify", "error", "CLI execution did not succeed.")
            raise RuntimeError("Unable to generate Graphify graph")
