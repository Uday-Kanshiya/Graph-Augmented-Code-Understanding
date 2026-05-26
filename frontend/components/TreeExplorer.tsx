"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import type { TreeNode, TreeSitterDocument } from "@/lib/types";

export function TreeExplorer({ document }: { document?: TreeSitterDocument | null }) {
  const [highlight, setHighlight] = useState<[number, number] | null>(null);

  if (!document) {
    return <div className="border border-dashed border-line bg-white p-6 text-sm text-zinc-500">Select a Python file to inspect its Tree-sitter parse tree.</div>;
  }

  const lines = document.source.split(/\r?\n/);

  return (
    <div className="grid min-h-[520px] grid-cols-1 gap-3 lg:grid-cols-[420px_1fr]">
      <div className="max-h-[620px] overflow-auto border border-line bg-white p-3 scrollbar-thin">
        {document.parse_error ? (
          <p className="text-sm text-rust">{document.parse_error}</p>
        ) : document.root ? (
          <TreeBranch node={document.root} depth={0} onHover={setHighlight} />
        ) : (
          <p className="text-sm text-zinc-500">No tree output available.</p>
        )}
      </div>
      <div className="max-h-[620px] overflow-auto border border-line bg-[#fbfcfa] p-3 font-mono text-xs leading-5 scrollbar-thin">
        {lines.map((line, index) => {
          const lineNumber = index + 1;
          const active = highlight && lineNumber >= highlight[0] && lineNumber <= highlight[1];
          return (
            <div key={lineNumber} className={active ? "bg-[#dceeea]" : ""}>
              <span className="mr-4 inline-block w-10 select-none text-right text-zinc-400">{lineNumber}</span>
              <span>{line || " "}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TreeBranch({
  node,
  depth,
  onHover,
}: {
  node: TreeNode;
  depth: number;
  onHover: (range: [number, number] | null) => void;
}) {
  const [open, setOpen] = useState(depth < 3);
  const hasChildren = node.children.length > 0;
  const startLine = node.start_point[0] + 1;
  const endLine = node.end_point[0] + 1;

  return (
    <div>
      <button
        type="button"
        className="flex w-full items-center gap-1 py-0.5 text-left text-xs hover:bg-panel"
        style={{ paddingLeft: depth * 12 }}
        onClick={() => setOpen((value) => !value)}
        onMouseEnter={() => onHover([startLine, endLine])}
        onMouseLeave={() => onHover(null)}
      >
        {hasChildren ? open ? <ChevronDown size={13} /> : <ChevronRight size={13} /> : <span className="w-[13px]" />}
        <span className={node.named ? "font-semibold text-ink" : "text-zinc-500"}>{node.type}</span>
        <span className="text-zinc-400">
          {startLine}:{node.start_point[1]}-{endLine}:{node.end_point[1]}
        </span>
        {node.text_preview ? <span className="truncate text-zinc-500">{node.text_preview}</span> : null}
      </button>
      {open && hasChildren ? node.children.map((child, index) => <TreeBranch key={`${child.type}-${child.start_byte}-${index}`} node={child} depth={depth + 1} onHover={onHover} />) : null}
    </div>
  );
}
