from __future__ import annotations

from dataclasses import dataclass

import json

from src.config.config import config
from src.ingestion.indexer import get_collection

@dataclass
class Augment:
    text: str
    source: str
    title: str
    heading: str
    tags: list[str]
    score: float

def _parse_tags(raw_tags: str) -> list[str]:
    try:
        parsed = json.loads(raw_tags)
        if isinstance(parsed, list):
            return [str(tag) for tag in parsed]
    except (TypeError, json.JSONDecodeError):
        pass
    return []


def search(query: str, top_k: int | None = None) -> list[Augment]:
    k = top_k or config.top_k
    collection = get_collection()
    total_docs = collection.count()

    if total_docs == 0:
        return []
    
    results = collection.query(
        query_texts=[query],
        n_results=min(k, total_docs),
        include=["documents", "metadatas", "distances"],
    )

    output: list[Augment] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        score = 1.0 - dist
        if score < config.similarity_threshold:
            continue
        
        output.append(
            Augment(
                text=doc,
                source=meta.get("source", ""),
                title=meta.get("title", ""),
                heading=meta.get("heading", ""),
                tags=_parse_tags(meta.get("tags", "[]")),
                score=score,
            )
        )

    return output
