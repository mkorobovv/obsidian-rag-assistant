from __future__ import annotations

import hashlib
import json

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from src.ingestion.loader import Chunk
from src.config.config import config

def _chunk_id(chunk: Chunk) -> str:
    key = f"{chunk.source}::{chunk.chunk_index}"
    return hashlib.md5(key.encode()).hexdigest()

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=config.db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.embedding_model
    )
    return client.get_or_create_collection(
        name=config.db_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

def index_chunks(chunks: list[Chunk], batch_size: int = 64) -> int:
    collection = get_collection()

    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if _chunk_id(c) not in existing_ids]

    if not new_chunks:
        return 0

    added = 0
    for i in tqdm(range(0, len(new_chunks), batch_size), desc="Indexing", unit="batch"):
        batch = new_chunks[i : i + batch_size]
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
        added += len(batch)

    return added


def collection_stats() -> dict:
    col = get_collection()
    count = col.count()
    return {"total_chunks": count, "collection": config.db_name}