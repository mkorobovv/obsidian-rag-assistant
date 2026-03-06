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

    def format_citation(self) -> str:
        parts = [f"📄 **{self.title}**"]
        if self.heading:
            parts.append(f"› {self.heading}")
        parts.append(f"  `{self.source}`")
        return "  ".join(parts)

def search(query: str, top_k: int | None = None) -> list[Augment]:
    k = top_k or config.top_k
    collection = get_collection()

    if collection.count() == 0:
        return []
    
    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
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
                tags=json.loads(meta.get("tags", "[]")),
                score=score,
            )
        )

    return output