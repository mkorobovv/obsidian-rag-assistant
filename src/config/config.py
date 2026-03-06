from dataclasses import dataclass
import os

@dataclass
class Config:
    vault_path: str = os.getenv("OBSIDIAN_VAULT_PATH", "")
    db_path: str = os.getenv("DB_PATH", "./chroma_db")
    db_name: str = "obsidian_notes"
    embedding_model: str = "intfloat/multilingual-e5-small"

    ## LLM
    api_token: str = os.getenv("API_TOKEN", "")
    provider: str = os.getenv("PROVIDER", "hf-inference")
    max_tokens: int = 256
    llm_model: str = os.getenv("LLM_MODEL", "CohereLabs/tiny-aya-global:cohere")

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5          # number of chunks returned per query
    similarity_threshold: float = 0.3

    # CLI
    show_sources: bool = True
    max_history: int = 20   # conversation turns kept in context

config = Config()