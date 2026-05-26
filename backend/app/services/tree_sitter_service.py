from __future__ import annotations

from pathlib import Path

from app.models.schemas import TreeNode, TreeSitterDocument
from app.services.file_utils import read_text_lossy


class TreeSitterService:
    def __init__(self) -> None:
        self._parser = None

    def parse_file(self, repo_id: str, repo_root: Path, file_path: str) -> TreeSitterDocument:
        source_path = repo_root / file_path
        source = read_text_lossy(source_path)
        try:
            parser = self._get_parser()
            tree = parser.parse(source.encode("utf-8"))
            root = self._serialize_node(tree.root_node, source.encode("utf-8"))
            warnings = ["Tree-sitter reported syntax errors."] if tree.root_node.has_error else []
            return TreeSitterDocument(repo_id=repo_id, file_path=file_path, source=source, root=root, warnings=warnings)
        except Exception as exc:
            return TreeSitterDocument(repo_id=repo_id, file_path=file_path, source=source, parse_error=str(exc))

    def _get_parser(self):
        if self._parser is not None:
            return self._parser
        from tree_sitter import Language, Parser
        import tree_sitter_python

        language = Language(tree_sitter_python.language())
        parser = Parser()
        try:
            parser.language = language
        except AttributeError:  # pragma: no cover - older tree-sitter API
            parser.set_language(language)
        self._parser = parser
        return parser

    def _serialize_node(self, node, source_bytes: bytes, max_preview: int = 120) -> TreeNode:
        text_preview = None
        if node.child_count == 0 and node.end_byte > node.start_byte:
            text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            text_preview = text.replace("\n", "\\n")[:max_preview]
        return TreeNode(
            type=node.type,
            named=node.is_named,
            start_point=tuple(node.start_point),
            end_point=tuple(node.end_point),
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            text_preview=text_preview,
            children=[self._serialize_node(child, source_bytes) for child in node.children],
        )

