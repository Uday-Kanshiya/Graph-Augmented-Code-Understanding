from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import ModelInfo, TokenMeasurement


class LLMConfigurationError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: TokenMeasurement
    response_tokens: TokenMeasurement
    total_tokens: TokenMeasurement


class LLMProvider:
    def count_tokens(self, text: str, stage: str = "llm_prompt_tokens") -> TokenMeasurement:
        raise NotImplementedError

    def generate_answer(self, prompt: str) -> LLMResponse:
        raise NotImplementedError

    def get_model_info(self) -> ModelInfo:
        raise NotImplementedError

