from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from huggingface_hub import InferenceClient

from src.config.config import config
from src.retrieval.search import Augment, search

import os

SYSTEM_PROMPT = """
Ты — персональный ассистент для работы с заметками Obsidian.
Отвечай ТОЛЬКО на основе предоставленного контекста из заметок.
Если ответа в контексте нет — так и скажи: "В заметках нет информации об этом."
Не придумывай ответы из общих знаний. Отвечай на том же языке, что и вопрос.
"""


def _build_context_block(results: list[Augment]) -> str:
    if not results:
        return "No relevant notes found."
    parts = []
    for i, r in enumerate(results, 1):
        heading_line = f" — {r.heading}" if r.heading else ""
        parts.append(
            f"[{i}] **{r.title}**{heading_line} (score: {r.score:.2f})\n{r.text}"
        )
    return "\n\n---\n\n".join(parts)

@dataclass
class Chain:
    _client: InferenceClient = field(init=False)
    _history: list[dict] = field(default_factory=list, init=False)
    _provider: str = field(init=False)
    _model: str = field(init=False)

    def __post_init__(self):
        provider = config.llm_provider.strip().lower()
        self._provider = provider
        self._model = config.llm_model

        if provider == "hf":
            token = (
                config.api_token
                or os.getenv("API_TOKEN", "")
                or os.getenv("HF_TOKEN", "")
            )
            if not token:
                raise EnvironmentError("Set API_TOKEN or HF_TOKEN for LLM_PROVIDER=hf.")
            self._client = InferenceClient(
                api_key=token,
            )
            return

        if provider == "openai":
            token = config.openai_api_key or os.getenv("OPENAI_API_KEY", "")
            if not token:
                raise EnvironmentError("Set OPENAI_API_KEY for LLM_PROVIDER=openai.")
            self._client = InferenceClient(
                provider="openai",
                api_key=token,
            )
            return

        raise EnvironmentError("Unsupported LLM_PROVIDER. Use one of: hf, openai.")

    def _trim_history(self):
        max_msgs = config.max_history * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def chat(
        self,
        user_query: str,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, list[Augment]]:
        results = search(user_query)

        context = _build_context_block(results)
        augmented_message = (
            f"<context>\n{context}\n</context>\n\n"
            f"Question: {user_query}"
        )

        self._history.append({"role": "user", "content": augmented_message})
        self._trim_history()

        full_response = ""
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *self._history,
            ],
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta and on_token:
                on_token(delta)
            full_response += delta

        self._history.append({"role": "assistant", "content": full_response})

        return full_response, results

    def reset(self):
        self._history.clear()
