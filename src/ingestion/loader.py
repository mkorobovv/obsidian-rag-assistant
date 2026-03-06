from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from src.config.config import config

@dataclass
class Chunk:
    text: str
    source: str
    title: str
    chunk_index: int
    tags: list[str]
    heading: str = ""

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_TAG_RE = re.compile(r"(?:^|\s)#([a-zA-Z0-9_/-]+)")
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)", re.MULTILINE)

def _strip_frontmatter(text: str) -> tuple[str, list[str]]:
    tags: list[str] = []

    match = _FRONTMATTER_RE.match(text)
    if match:
        front = match.group(0)
        for line in front.splitlines():
            if line.strip().startswith("tags:"):
                raw = line.split(":", 1)[1]
                tags.extend(t.strip().strip('"').strip("'") for t in re.split(r"[,\[\]]", raw) if t.strip())
        text = text[match.end():]

    tags += _TAG_RE.findall(text)

    return text.strip(), list(set(tags))

def _nearest_heading(text: str, pos: int) -> str:
    heading = ""
    for m in _HEADING_RE.finditer(text):
        if m.start() > pos:
            break
        heading = m.group(1)
    return heading

def _split_into_chunks(text: str, size: int, overlap: int) -> Generator[tuple[str, int], None, None]:
    words = text.split()
    step = max(1, size - overlap)
    i = 0
    pos = 0
    while i < len(words):
        chunk_words = words[i : i + size]
        yield " ".join(chunk_words), pos
        pos += len(" ".join(words[i : i + step]))
        i += step

def get_chunks(vault_path: str | None = None) -> list[Chunk]:
    root = Path(vault_path or config.vault_path)

    if not root.exists():
        raise FileNotFoundError(f"Vault path {root} does not exist")
    
    chunks: list[Chunk] = []
    md_files = list(root.rglob("*.md"))

    if not md_files:
        raise ValueError(f"No .md files found in {root}")
    
    for path in md_files:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        text, tags = _strip_frontmatter(raw)
        if not text.strip():
            continue

        relative = str(path.relative_to(root))
        title = path.stem

        for idx, (chunk_text, start_pos) in enumerate(
            _split_into_chunks(text, config.chunk_size, config.chunk_overlap)
        ):
            if not chunk_text.strip():
                continue

            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=relative,
                    title=title,
                    chunk_index=idx,
                    tags=tags,
                    heading=_nearest_heading(text, start_pos),
                )
            )
        
    return chunks