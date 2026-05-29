from __future__ import annotations

from pathlib import Path

from app.models.schemas import TreeNode, TreeSitterDocument
from app.services.file_utils import read_text_lossy


class TreeSitterService:
    def __init__(self) -> None:
        self._parsers = {}

    def parse_file(self, repo_id: str, repo_root: Path, file_path: str, language_name: str = "python") -> TreeSitterDocument:
        source_path = repo_root / file_path
        source = read_text_lossy(source_path)
        try:
            parser = self._get_parser(language_name)
            if not parser:
                raise ValueError(f"Tree-sitter parser not available for language: {language_name}")
            tree = parser.parse(source.encode("utf-8"))
            root = self._serialize_node(tree.root_node, source.encode("utf-8"))
            warnings = ["Tree-sitter reported syntax errors."] if tree.root_node.has_error else []
            return TreeSitterDocument(repo_id=repo_id, file_path=file_path, language=language_name, source=source, root=root, warnings=warnings)
        except Exception as exc:
            return TreeSitterDocument(repo_id=repo_id, file_path=file_path, language=language_name, source=source, parse_error=str(exc))

    def _get_parser(self, language_name: str):
        if language_name in self._parsers:
            return self._parsers[language_name]

        from tree_sitter import Language, Parser

        try:
            if language_name == "python":
                import tree_sitter_python as lang_module
                lang_func = lang_module.language
            elif language_name == "javascript":
                import tree_sitter_javascript as lang_module
                lang_func = lang_module.language
            elif language_name == "typescript":
                import tree_sitter_typescript as lang_module
                if hasattr(lang_module, "language_typescript"):
                    lang_func = lang_module.language_typescript
                elif hasattr(lang_module, "language"):
                    lang_func = lang_module.language
                else:
                    lang_func = lang_module.language_tsx
            elif language_name == "go":
                import tree_sitter_go as lang_module
                lang_func = lang_module.language
            elif language_name == "rust":
                import tree_sitter_rust as lang_module
                lang_func = lang_module.language
            elif language_name == "java":
                import tree_sitter_java as lang_module
                lang_func = lang_module.language
            elif language_name == "cpp":
                import tree_sitter_cpp as lang_module
                lang_func = lang_module.language
            elif language_name == "c":
                import tree_sitter_c as lang_module
                lang_func = lang_module.language
            else:
                return None

            language = Language(lang_func())
            parser = Parser()
            try:
                parser.language = language
            except AttributeError:
                parser.set_language(language)
            self._parsers[language_name] = parser
            return parser
        except (ImportError, AttributeError, Exception):
            # Fallback to python parser if the requested language module is not installed or raises errors
            try:
                import tree_sitter_python as lang_module
                language = Language(lang_module.language())
                parser = Parser()
                try:
                    parser.language = language
                except AttributeError:
                    parser.set_language(language)
                self._parsers[language_name] = parser
                return parser
            except Exception:
                return None


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

