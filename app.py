from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config.config import config

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════╗
║         Obsidian RAG Assistant           ║
╚══════════════════════════════════════════╝{RESET}
Type {GREEN}/help{RESET} for commands, {GREEN}/quit{RESET} to exit.
""")
    
def print_sources(results):
    if not results:
        return
    print(f"\n{DIM}── Sources ────────────────────────────────────{RESET}")
    for r in results:
        print(f"{DIM}{r.format_citation()}  [{r.score:.0%}]{RESET}")
    print(f"{DIM}───────────────────────────────────────────────{RESET}\n")

HELP_TEXT = f"""
{BOLD}Commands:{RESET}
  {GREEN}/help{RESET}           Show this message
  {GREEN}/sources on|off{RESET} Toggle source citations (currently {{}})
  {GREEN}/reset{RESET}          Clear conversation history
  {GREEN}/reindex{RESET}        Re-index vault (picks up new/changed notes)
  {GREEN}/stats{RESET}          Show index statistics
  {GREEN}/vault <path>{RESET}   Change vault path and re-index
  {GREEN}/quit{RESET}           Exit
"""

def do_index(vault_path: str | None = None, force: bool = False):
    from src.ingestion.loader import get_chunks
    from src.ingestion.indexer import index_chunks, collection_stats

    vpath = vault_path or config.vault_path
    print(f"{YELLOW}📂 Loading notes from:{RESET} {vpath}")
    try:
        chunks = get_chunks(vpath)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌  {e}")
        sys.exit(1)

    print(f"{YELLOW}✨ Found {len(chunks)} chunks across {len({c.source for c in chunks})} notes{RESET}")
    added = index_chunks(chunks)
    stats = collection_stats()

    if added:
        print(f"{GREEN}✅  Indexed {added} new chunks → {stats['total_chunks']} total in ChromaDB{RESET}\n")
    else:
        print(f"{GREEN}✅  Index up to date — {stats['total_chunks']} chunks in ChromaDB{RESET}\n")

def main():
    parser = argparse.ArgumentParser(description="Obsidian RAG Assistant")
    parser.add_argument("--vault", help="Path to Obsidian vault")
    parser.add_argument("--reindex", action="store_true", help="Force re-index")
    args = parser.parse_args()

    banner()

    do_index(force=args.reindex)

    from src.chain.chain import Chain
    try:
        chain = Chain()
    except EnvironmentError as e:
        print(f"❌  {e}")
        sys.exit(1)

    show_sources = config.show_sources

    while True:
        try:
            user_input = input(f"{GREEN}Input:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd_parts = user_input.lower().split()
            cmd = cmd_parts[0]
            if cmd == "/quit":
                break
            elif cmd == "/reset":
                chain.reset() 
            elif cmd == "/stats":
                from src.ingestion.indexer import collection_stats
                s = collection_stats()
                print(f"  Chunks in index: {s['total_chunks']}")
            elif cmd == "/reindex":
                do_index(force=True)
            elif cmd == "/vault" and len(cmd_parts) > 1:
                new_path = cmd_parts[1]
                config.vault_path = str(Path(new_path).expanduser().resolve())
                do_index(vault_path=config.vault_path, force=True)
            else:
                print(f"  Unknown command. Type /help for options.")
            continue
            
        print(f"\n{CYAN}Assistant:{RESET} ", end="", flush=True)
        try:
            _, results = chain.chat(user_input)
        except Exception as e:
            print(f"\n❌  Error: {e}")
            continue

        if show_sources:
            print_sources(results)

if __name__ == "__main__":
    main()