# Obsidian RAG Assistant

CLI RAG assistant for Obsidian vaults (`.md` files).  
It indexes your notes into ChromaDB, retrieves relevant chunks for each question, and sends context to an Inference chat model.

## Features

- Indexes all Markdown files in an Obsidian vault.
- Extracts tags and nearest headings for better context.
- Uses local vector store (`ChromaDB`) with sentence-transformer embeddings.
- Streams assistant responses in CLI.
- Optional source citations after each answer.
- Supports incremental re-index and force re-index.

## Requirements

- Hugging Face token with access to your selected model
- Obsidian vault path with `.md` files

## Installation

```bash
git clone https://github.com/mkorobovv/obsidian-rag-assistant.git
cd obsidian-rag-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create `.env` in the project root:

```env
# Required
VAULT_PATH=/absolute/path/to/your/obsidian/vault
LLM_PROVIDER=hf

# Required if LLM_PROVIDER=hf
HF_TOKEN=hf_xxx

# Required if LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx

# Optional
DB_PATH=./chroma_db
LLM_MODEL=CohereLabs/tiny-aya-global:cohere
```

Notes:

- `LLM_PROVIDER` supports `hf` and `openai`.
- `API_TOKEN` and `HF_TOKEN` are alternatives for HF.
- For OpenAI provider, use an OpenAI model in `LLM_MODEL` (for example `gpt-4o-mini`).
- `DB_PATH` defaults to `./chroma_db`.

## Run

```bash
python app.py
```

Optional flags:

```bash
python app.py --vault /absolute/path/to/vault
python app.py --reindex
python app.py --vault /absolute/path/to/vault --reindex
```

Startup behavior:

1. Loads config.
2. Indexes vault content.
3. Starts interactive chat loop.

## CLI Commands

- `/help` Show command list
- `/sources on|off` Toggle source citations
- `/reset` Clear chat history
- `/reindex` Force re-index (also cleans stale chunks)
- `/stats` Show chunk count in index
- `/vault <path>` Switch vault and re-index
- `/clear` Clear terminal and redraw header
- `/quit` Exit

## How Retrieval Works

1. Vault loader reads all `*.md`.
2. Frontmatter is stripped; tags are extracted from frontmatter and inline hashtags.
3. Notes are split into overlapping word chunks.
4. Chunks are embedded with `intfloat/multilingual-e5-small` (default).
5. Chunks are stored in Chroma collection `obsidian_notes`.
6. On each question, top `k` chunks are retrieved and filtered by similarity threshold.
7. Retrieved context is injected into the prompt and sent to the chat model.

Current defaults (from code):

- Chunk size: `512` words
- Chunk overlap: `64` words
- Top-k: `3`
- Similarity threshold: `0.3`
- Max chat history turns: `20`

## Project Structure

```text
.
├── app.py                    # CLI entrypoint
├── requirements.txt
└── src/
    ├── chain/chain.py        # Chat pipeline
    ├── config/config.py      # Environment and defaults
    ├── ingestion/loader.py   # Vault parsing + chunking
    ├── ingestion/indexer.py  # Chroma collection + indexing
    └── retrieval/search.py   # Vector search
```
