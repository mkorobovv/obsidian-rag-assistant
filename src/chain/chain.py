from __future__ import annotations

from dataclasses import dataclass, field

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

    def __post_init__(self):
        if not config.api_token:
            raise EnvironmentError("API_TOKEN is not set.")
        self._client = InferenceClient(
            api_key=os.environ["HF_TOKEN"],
        )

    def _trim_history(self):
        max_msgs = config.max_history * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def chat(self, user_query: str) -> tuple[str, list[Augment]]:
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
            model=config.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *self._history,
            ],
            stream=True,
            max_tokens=config.max_tokens,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_response += delta

        print()

        self._history.append({"role": "assistant", "content": full_response})

        return full_response, results

    def reset(self):
        self._history.clear()
