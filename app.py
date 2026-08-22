"""CLI entry point for the document reviewer agent.

Two commands:
  python app.py ingest [directory]   load docs, split, embed, persist
  python app.py ask                  interactive Q&A loop over the docs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from document_reviewer.config.settings import settings
from document_reviewer.loader import load_documents
from document_reviewer.rag_chain import build_rag_chain
from document_reviewer.splitter import split_documents
from document_reviewer.tools import STRATEGY_NAMES
from document_reviewer.vector_store import build_vector_store


def ingest(directory: str, strategy: str = "vector", force: bool = False) -> None:
    docs = load_documents(directory)
    if not docs:
        print(f"No supported documents found in '{directory}' (looking for .txt, .md).")
        return

    chunks = split_documents(docs)

    if strategy == "contextual":
        from document_reviewer.tools.contextual import build_vector_store_contextual

        print(
            f"Contextual ingestion makes one LLM call per chunk: {len(chunks)} chunk(s) "
            f"-> {len(chunks)} call(s)."
        )
        build_vector_store_contextual(docs, chunks, force=force)
    elif strategy == "graphrag":
        from document_reviewer.tools.graphrag import build_vector_store_graphrag

        if not force:
            print(
                f"GraphRAG ingestion makes one LLM call per chunk for entity/relation "
                f"extraction ({len(chunks)} chunk(s) -> {len(chunks)} call(s)), and is the "
                "most expensive strategy. Pass --force to confirm and proceed."
            )
            return
        build_vector_store_graphrag(docs, chunks)
    else:
        build_vector_store(chunks)

    print(f"Ingested {len(docs)} document(s) -> {len(chunks)} chunk(s) from '{directory}' (strategy: {strategy}).")


def ask(strategy: str | None = None, mode: str = "manual") -> None:
    """mode="manual" uses `strategy` (or settings.retrieval_strategy)
    for every question. mode="automated" re-routes per question instead,
    among whichever strategies actually have an index built — see
    tools.get_available_strategies and router.Router.
    """
    chain = None
    router = None
    available: list[str] = []

    if mode == "automated":
        from document_reviewer.router import get_router
        from document_reviewer.tools import get_available_strategies

        available = get_available_strategies()
        if not available:
            print("No ingested strategy found. Run `python app.py ingest` first.")
            return
        router = get_router(settings.router_type)
        print(f"Automated mode ({settings.router_type} router): routing each question among {available}.")
    else:
        chain = build_rag_chain(strategy=strategy)
        print(f"Using retrieval strategy: {strategy or settings.retrieval_strategy}")

    print("Ask a question about your documents. Type 'exit' or Ctrl+D to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        if router is not None:
            selected_strategy = router.select_strategy(question, available)
            print(f"[router selected: {selected_strategy}]")
            active_chain = build_rag_chain(strategy=selected_strategy)
        else:
            active_chain = chain

        result = active_chain.invoke(question)
        sources = sorted({Path(d.metadata["source"]).name for d in result["source_documents"]})

        print(f"\n{result['answer']}")
        print(f"(retrieved from: {', '.join(sources)})\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Document reviewer RAG agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load, chunk, and index a directory of documents.")
    ingest_parser.add_argument("directory", nargs="?", default="data", help="Directory to ingest (default: data)")
    ingest_parser.add_argument(
        "--strategy",
        choices=STRATEGY_NAMES,
        default="vector",
        help="Ingestion path to use — most strategies share the default index; "
        "'contextual' builds a separate one (default: vector)",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="For --strategy contextual: proceed even if chunk count exceeds the safety limit. "
        "For --strategy graphrag: required to confirm the run (always expensive).",
    )

    ask_parser = subparsers.add_parser("ask", help="Interactively ask questions about ingested documents.")
    ask_parser.add_argument(
        "--strategy",
        choices=STRATEGY_NAMES,
        default=None,
        help="Retrieval strategy to use (default: from settings). Mutually exclusive with --mode automated.",
    )
    ask_parser.add_argument(
        "--mode",
        choices=["manual", "automated"],
        default="manual",
        help="'manual' (default) uses --strategy for every question; "
        "'automated' routes each question to a strategy automatically",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.directory, strategy=args.strategy, force=args.force)
    elif args.command == "ask":
        if args.mode == "automated" and args.strategy is not None:
            parser.error("--mode automated and --strategy are mutually exclusive.")
        ask(strategy=args.strategy, mode=args.mode)


if __name__ == "__main__":
    main()
