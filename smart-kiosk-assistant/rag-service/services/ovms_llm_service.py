"""OVMS-backed LLM service.

Provides the same public interface as ``LLMService`` (generate, generate_stream,
count_tokens, close, as_text_generator, .tokenizer) but delegates inference to
the OVMS OpenAI-compatible endpoint instead of loading a model in-process.

This eliminates the duplicate GPU load: a single Qwen3-4B instance served by
OVMS is shared by both the ordering agent (ADK / LiteLlm) and the RAG pipeline.
"""
from __future__ import annotations

import logging
import time
from typing import Generator

from openai import OpenAI
from transformers import AutoTokenizer

from agentic import config as agent_cfg
from utils.latency_store import llm_latency

logger = logging.getLogger(__name__)


class OVMSLLMService:
    """LLM service backed by OVMS /v3/chat/completions.

    Args:
        hf_id: HuggingFace model id used *only* to load the tokenizer locally
               (for count_tokens and SemanticChunker size splitting). The model
               itself runs inside OVMS.
        temperature: Default sampling temperature (0.0 = greedy).
        default_max_new_tokens: Default token budget for completions.
        generation_timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        hf_id: str,
        temperature: float = 0.0,
        default_max_new_tokens: int = 192,
        generation_timeout: float = 90.0,
    ) -> None:
        self._model_id = agent_cfg.LLM_MODEL
        self._base_url = agent_cfg.LLM_URL
        self._temperature = temperature
        self._default_max_new_tokens = default_max_new_tokens
        self._timeout = generation_timeout

        logger.info(
            "[OVMS-LLM] Connecting to OVMS base_url=%s model=%s",
            self._base_url, self._model_id,
        )
        self._client = OpenAI(base_url=self._base_url, api_key="unused")

        logger.info("[OVMS-LLM] Loading tokenizer from HF id=%s (local only)", hf_id)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_id, fix_mistral_regex=True)
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_id)

    # ── public API (mirrors LLMService) ─────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self._temperature
        max_tok = max_tokens if max_tokens is not None else self._default_max_new_tokens
        prompt_tokens = self.count_tokens(prompt)
        _t0 = time.monotonic()

        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tok,
            temperature=temp,
            timeout=self._timeout,
        )
        text: str = response.choices[0].message.content or ""

        dt = time.monotonic() - _t0
        completion_tokens = self.count_tokens(text)
        tps = (completion_tokens / dt) if dt > 0 else 0.0
        logger.info(
            "[OVMS-LLM] generate prompt_tokens=%d completion_tokens=%d elapsed=%.2fs tps=%.1f",
            prompt_tokens, completion_tokens, dt, tps,
        )
        llm_latency.record(dt * 1000)
        return text

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        temp = temperature if temperature is not None else self._temperature
        max_tok = max_tokens if max_tokens is not None else self._default_max_new_tokens
        _t0 = time.monotonic()

        try:
            with self._client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tok,
                temperature=temp,
                stream=True,
                timeout=self._timeout,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
        finally:
            llm_latency.record((time.monotonic() - _t0) * 1000)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001
            return max(1, len(text) // 4)

    def close(self) -> None:
        """No persistent resources to release (OVMS manages the model)."""

    def as_text_generator(self):
        """Returns ``generate`` bound as a plain callable for SemanticChunker."""
        return self.generate
