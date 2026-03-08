from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config.config import config

LOGGER = logging.getLogger(__name__)

COLOR_RESET = "\033[0m"
COLOR_FAINT = "\033[2m"
COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_BLUE = "\033[34m"

HELP_TEXT = """
Commands:
  /help           Show this message
  /sources on|off Toggle source citations (currently {})
  /reset          Clear conversation history
  /reindex        Re-index vault (picks up new/changed notes)
  /stats          Show index statistics
  /vault <path>   Change vault path and re-index
  /clear          Clear terminal and redraw header
  /quit           Exit
"""


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for logger_name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def banner() -> None:
    print(f"{COLOR_CYAN}Obsidian RAG Assistant{COLOR_RESET}")
    print(f"{COLOR_FAINT}Type /help for commands, /quit to exit.{COLOR_RESET}\n")


def print_session_status() -> None:
    print(f"{COLOR_FAINT}Session{COLOR_RESET}")
    print(f"  Provider: {config.llm_provider}")
    print(f"  Model:   {config.llm_model}")
    print(f"  Vault:   {config.vault_path}")
    print(f"  Sources: {'on' if config.show_sources else 'off'}\n")


def print_sources(results: list) -> None:
    if not results:
        return
    print(f"\n{COLOR_BLUE}Sources{COLOR_RESET}")
    for r in results:
        print(f"- {r.title} [{r.score:.0%}]")
        print(f"  {COLOR_FAINT}{r.source}{COLOR_RESET}")
    print()

def stream_stdout(text: str) -> None:
    print(text, end="", flush=True)


def do_index(vault_path: str | None = None, force: bool = False) -> None:
    from src.ingestion.loader import get_chunks
    from src.ingestion.indexer import index_chunks, collection_stats

    vpath = vault_path or config.vault_path
    print(f"{COLOR_FAINT}Indexing vault:{COLOR_RESET} {vpath}")
    try:
        chunks = get_chunks(vpath)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Found {len(chunks)} chunks across {len({c.source for c in chunks})} notes")
    try:
        result = index_chunks(chunks, force=force)
        stats = collection_stats()
    except Exception as e:
        LOGGER.exception("Indexing failed")
        print(f"Error: indexing failed: {e}")
        print("    Ensure network access for downloading embedding models, or use a pre-cached local model.")
        sys.exit(1)
    upserted = result["upserted"]
    deleted = result["deleted"]
    if force:
        print(f"{COLOR_GREEN}Done.{COLOR_RESET} Re-indexed {upserted} chunks, removed {deleted} stale chunks.")
        print(f"Total chunks in ChromaDB: {stats['total_chunks']}\n")
    else:
        print(f"{COLOR_GREEN}Done.{COLOR_RESET} Indexed/updated {upserted} chunks.")
        print(f"Total chunks in ChromaDB: {stats['total_chunks']}\n")


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


def handle_command(
    user_input: str,
    chain,
    show_sources: bool,
) -> tuple[bool, bool]:
    cmd_parts = user_input.split()
    cmd = cmd_parts[0].lower()

    if cmd == "/quit":
        return False, show_sources

    if cmd == "/help":
        state = "on" if show_sources else "off"
        print(HELP_TEXT.format(state))
        return True, show_sources

    if cmd == "/clear":
        clear_screen()
        banner()
        print_session_status()
        return True, show_sources

    if cmd == "/sources":
        if len(cmd_parts) < 2 or cmd_parts[1].lower() not in {"on", "off"}:
            print("  Usage: /sources on|off")
            return True, show_sources
        show_sources = cmd_parts[1].lower() == "on"
        print(f"  Source citations are now {'on' if show_sources else 'off'}.")
        return True, show_sources

    if cmd == "/reset":
        chain.reset()
        return True, show_sources

    if cmd == "/stats":
        from src.ingestion.indexer import collection_stats

        stats = collection_stats()
        print(f"  Chunks in index: {stats['total_chunks']}")
        return True, show_sources

    if cmd == "/reindex":
        do_index(force=True)
        return True, show_sources

    if cmd == "/vault":
        if len(cmd_parts) < 2:
            print("  Usage: /vault <path>")
            return True, show_sources
        new_path = " ".join(cmd_parts[1:])
        config.vault_path = str(Path(new_path).expanduser().resolve())
        do_index(vault_path=config.vault_path, force=True)
        return True, show_sources

    print("  Unknown command. Type /help for options.")
    return True, show_sources


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Obsidian RAG Assistant")
    parser.add_argument("--vault", help="Path to Obsidian vault")
    parser.add_argument("--reindex", action="store_true", help="Force re-index")
    args = parser.parse_args()

    banner()
    if args.vault:
        config.vault_path = str(Path(args.vault).expanduser().resolve())
    print_session_status()

    do_index(vault_path=config.vault_path, force=args.reindex)

    from src.chain.chain import Chain
    try:
        chain = Chain()
    except EnvironmentError as e:
        LOGGER.error("LLM configuration error: %s", e)
        print(f"Error: {e}")
        sys.exit(1)

    show_sources = config.show_sources

    while True:
        try:
            user_input = input(f"{COLOR_CYAN}you{COLOR_RESET} {COLOR_FAINT}>{COLOR_RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            should_continue, show_sources = handle_command(user_input, chain, show_sources)
            if not should_continue:
                break
            continue

        print(f"\n{COLOR_GREEN}assistant{COLOR_RESET} {COLOR_FAINT}>{COLOR_RESET} ", end="", flush=True)
        try:
            _, results = chain.chat(user_input, on_token=stream_stdout)
            print()
        except Exception as e:
            LOGGER.exception("Chat handling failed")
            print(f"\nError: {e}")
            continue

        if show_sources:
            print_sources(results)

if __name__ == "__main__":
    main()
