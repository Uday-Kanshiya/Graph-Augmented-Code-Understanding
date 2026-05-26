from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import CodeChunk, SourceSnippet, TokenMeasurement
from app.services.storage import LocalStorage
from app.services.token_service import TokenService


@dataclass
class RetrievalResult:
    context: str
    snippets: list[SourceSnippet]
    token_measurement: TokenMeasurement


class RetrievalService:
    def __init__(self, storage: LocalStorage, token_service: TokenService) -> None:
        self.storage = storage
        self.token_service = token_service

    def build_context(self, repo_id: str, query: str, limit: int = 8) -> RetrievalResult:
        chunks = self.storage.load_chunks(repo_id)
        if not chunks:
            raise ValueError("No chunks found for repo.")
        scored = sorted(
            ((self._score(chunk, query), chunk) for chunk in chunks),
            key=lambda item: item[0],
            reverse=True,
        )
        selected = [(score, chunk) for score, chunk in scored if score > 0][:limit]
        if not selected:
            selected = scored[: min(limit, len(scored))]
        snippets = [
            SourceSnippet(
                file_path=chunk.file_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                text=chunk.text,
                score=score,
                source="basic-retrieval",
            )
            for score, chunk in selected
        ]
        context = "\n\n".join(
            f"### {snippet.file_path}:{snippet.line_start}-{snippet.line_end}\n{snippet.text}"
            for snippet in snippets
        )
        measurement = self.token_service.measure_estimated("chunked_basic_retrieval_context", context)
        return RetrievalResult(context=context, snippets=snippets, token_measurement=measurement)

    def _score(self, chunk: CodeChunk, query: str) -> float:
        terms = [term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", query)]
        haystack = f"{chunk.file_path}\n{chunk.text}".lower()
        score = 0.0
        for term in terms:
            score += haystack.count(term)
            if term in chunk.file_path.lower():
                score += 5
        return score

