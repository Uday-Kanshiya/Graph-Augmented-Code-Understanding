from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.models.schemas import CodeChunk, CountType, GraphDocument, RepoFile, TokenMeasurement, TokenSummary
from app.services.file_utils import read_text_lossy


class TokenService:
    def __init__(self) -> None:
        self._encoding = None

    def estimate_tokens(self, text: str) -> int:
        try:
            if self._encoding is None:
                import tiktoken

                self._encoding = tiktoken.get_encoding("cl100k_base")
            return len(self._encoding.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def measure_estimated(self, stage: str, text: str, notes: str | None = None) -> TokenMeasurement:
        return TokenMeasurement(
            stage=stage,
            tokens=self.estimate_tokens(text),
            count_type=CountType.estimated,
            notes=notes or "Estimated with cl100k_base via tiktoken; Gemini exact counts are collected at LLM call time when configured.",
        )

    def build_chunks(self, repo_root: Path, files: list[RepoFile], lines_per_chunk: int = 80, overlap: int = 10) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for repo_file in files:
            text = read_text_lossy(repo_root / repo_file.path)
            lines = text.splitlines()
            if not lines:
                continue
            start = 0
            while start < len(lines):
                end = min(len(lines), start + lines_per_chunk)
                chunk_text = "\n".join(lines[start:end])
                raw_id = f"{repo_file.path}:{start + 1}:{end}:{hashlib.sha1(chunk_text.encode('utf-8')).hexdigest()[:8]}"
                chunks.append(
                    CodeChunk(
                        chunk_id=f"chunk:{hashlib.sha1(raw_id.encode('utf-8')).hexdigest()[:16]}",
                        file_path=repo_file.path,
                        line_start=start + 1,
                        line_end=end,
                        text=chunk_text,
                        token_estimate=self.estimate_tokens(chunk_text),
                    )
                )
                if end == len(lines):
                    break
                start = max(end - overlap, start + 1)
        return chunks

    def build_repo_summary(
        self,
        repo_id: str,
        repo_root: Path,
        files: list[RepoFile],
        chunks: list[CodeChunk],
        codegraph: GraphDocument,
        graphify: GraphDocument,
    ) -> TokenSummary:
        codegraph_text = json.dumps(codegraph.model_dump(mode="json"), separators=(",", ":"))
        graphify_text = json.dumps(graphify.model_dump(mode="json"), separators=(",", ":"))
        stages = {
            "codegraph_derived_context": self.measure_estimated("codegraph_derived_context", codegraph_text),
            "graphify_derived_context": self.measure_estimated("graphify_derived_context", graphify_text),
        }
        return TokenSummary(repo_id=repo_id, stages=stages)

