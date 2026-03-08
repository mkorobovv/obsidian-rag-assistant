from __future__ import annotations

import hashlib
import json

import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from tqdm import tqdm

from src.ingestion.loader import Chunk
from src.config.config import config


class QuietSentenceTransformerEmbeddingFunction(
    embedding_functions.SentenceTransformerEmbeddingFunction
):
    def __call__(self, input):
        embeddings = self._model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return [np.array(embedding, dtype=np.float32) for embedding in embeddings]


def _chunk_id(chunk: Chunk) -> str:
    key = f"{chunk.source}::{chunk.chunk_index}"
    return hashlib.md5(key.encode()).hexdigest()

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=config.db_path)
    ef = QuietSentenceTransformerEmbeddingFunction(
        model_name=config.embedding_model
    )
    return client.get_or_create_collection(
        name=config.db_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

def index_chunks(chunks: list[Chunk], batch_size: int = 64, force: bool = False) -> dict:
    collection = get_collection()
    existing_ids = set(collection.get(include=[])["ids"])
    current_ids = {_chunk_id(c) for c in chunks}

    upserted = 0
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing", unit="batch"):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[_chunk_id(c) for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "source": c.source,
                    "title": c.title,
                    "heading": c.heading,
                    "tags": json.dumps(c.tags),
                    "chunk_index": c.chunk_index,
                }
                for c in batch
            ],
        )
        upserted += len(batch)

    deleted = 0
    if force:
        stale_ids = list(existing_ids - current_ids)
        if stale_ids:
            for i in tqdm(range(0, len(stale_ids), batch_size), desc="Cleaning", unit="batch"):
                batch_ids = stale_ids[i : i + batch_size]
                collection.delete(ids=batch_ids)
                deleted += len(batch_ids)

    return {"upserted": upserted, "deleted": deleted}


def collection_stats() -> dict:
    col = get_collection()
    count = col.count()
    return {"total_chunks": count, "collection": config.db_name}
