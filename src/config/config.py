from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    vault_path: str = os.getenv("VAULT_PATH", "")
    db_path: str = os.getenv("DB_PATH", "./chroma_db")
    db_name: str = "obsidian_notes"
    embedding_model: str = "intfloat/multilingual-e5-small"

    ## LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "hf")
    api_token: str = os.getenv("HF_TOKEN", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    max_tokens: int = 256
    llm_model: str = os.getenv("LLM_MODEL", "")

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 3          # number of chunks returned per query
    similarity_threshold: float = 0.3

    # CLI
    show_sources: bool = True
    max_history: int = 20   # conversation turns kept in context

config = Config()
