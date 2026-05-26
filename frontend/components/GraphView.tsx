"use client";

import { Background, Controls, MiniMap, ReactFlow, type Edge, type Node } from "@xyflow/react";
import type { GraphDocument } from "@/lib/types";

const typeColors: Record<string, string> = {
  module: "#e6f4f1",
  class: "#f4e8df",
  function: "#e8eef6",
  method: "#eef0df",
  import: "#f3eafa",
  external_symbol: "#f4f4f4",
};

export function GraphView({ graph }: { graph?: GraphDocument | null }) {
  if (!graph) {
    return <div className="flex h-72 items-center justify-center border border-dashed border-line bg-white text-sm text-zinc-500">No graph loaded.</div>;
  }

  const nodes: Node[] = graph.nodes.slice(0, 150).map((node, index) => {
    const radius = 360;
    const angle = (index / Math.max(1, graph.nodes.length)) * Math.PI * 2;
    return {
      id: node.node_id,
      position: {
        x: Math.cos(angle) * radius + 420,
        y: Math.sin(angle) * radius + 320,
      },
      data: {
        label: `${node.node_type}: ${node.label}`,
      },
      style: {
        background: typeColors[node.node_type] ?? "#ffffff",
        borderColor: "#bfc8c2",
        width: 170,
      },
    };
  });

  const visible = new Set(nodes.map((node) => node.id));
  const edges: Edge[] = graph.edges
    .filter((edge) => visible.has(edge.source_node) && visible.has(edge.target_node))
    .slice(0, 300)
    .map((edge) => ({
      id: edge.edge_id,
      source: edge.source_node,
      target: edge.target_node,
      label: edge.edge_type,
      animated: edge.edge_type === "calls",
    }));

  return (
    <div className="h-[520px] overflow-hidden border border-line bg-white">
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <Background gap={18} size={1} color="#dfe3df" />
        <MiniMap pannable zoomable />
        <Controls />
      </ReactFlow>
    </div>
  );
}

