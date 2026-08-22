"""Registry of retrieval strategies.

Each "search tool" strategy module exposes two functions:
  search(vector_store, query) -> (formatted_context, source_documents)
      Plain function, called directly for manual dispatch and by the
      heuristic router.
  build_tool(vector_store) -> BaseTool
      Wraps `search` as a @tool, so it can be handed to llm.bind_tools()
      later (used by the agentic strategy and, eventually, an LLM router).

"Chain" strategies (agentic, graphrag) don't fit the one-query-in,
one-context-out shape and instead expose build_chain(vector_store) that
returns a full question -> {"answer", "source_documents"} runnable.

Most search-tool strategies share the default Chroma collection and
don't need their own get_vector_store — vector_store.get_vector_store()
covers them. A strategy whose embedded text differs from the raw chunk
text (contextual) exposes its own module-level get_vector_store() that
opens a different collection instead; get_vector_store_for_strategy()
below picks whichever applies.

All six strategies from PLAN.md are now wired up: "vector", "hybrid",
"rerank", "contextual", "agentic", "graphrag".
"""

from typing import Literal

from document_reviewer.tools import agentic, contextual, graphrag, hybrid, rerank, vector

STRATEGY_NAMES = ("vector", "hybrid", "rerank", "contextual", "agentic", "graphrag")
StrategyName = Literal["vector", "hybrid", "rerank", "contextual", "agentic", "graphrag"]

SEARCH_TOOL_STRATEGIES = {"vector", "hybrid", "rerank", "contextual"}
CHAIN_STRATEGIES = {"agentic", "graphrag"}

_SEARCH_MODULES = {"vector": vector, "hybrid": hybrid, "rerank": rerank, "contextual": contextual}
_CHAIN_MODULES = {"agentic": agentic, "graphrag": graphrag}


def get_search_module(name: str):
    if name not in _SEARCH_MODULES:
        raise ValueError(f"No search module registered for strategy '{name}'")
    return _SEARCH_MODULES[name]


def get_chain_module(name: str):
    if name not in _CHAIN_MODULES:
        raise ValueError(f"No chain module registered for strategy '{name}'")
    return _CHAIN_MODULES[name]


def get_vector_store_for_strategy(name: str):
    """Open whichever Chroma collection `name`'s strategy actually
    reads from. Falls back to the default collection for strategies
    that don't define their own get_vector_store (i.e. all of them
    except contextual, today)."""
    from document_reviewer.vector_store import get_vector_store as get_default_vector_store

    module = _SEARCH_MODULES.get(name) or get_chain_module(name)
    getter = getattr(module, "get_vector_store", get_default_vector_store)
    return getter()


def get_available_strategies() -> list[str]:
    """Which strategies actually have an index built on disk right
    now. This is the ONE place that constraint is checked — routers
    (heuristic today, LLM later) just receive this list and must
    never pick outside it; they don't re-derive it themselves.
    """
    from pathlib import Path

    import chromadb

    from document_reviewer.config import settings

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    existing_collections = {c.name for c in client.list_collections()}

    available = []
    if settings.chroma_collection_name in existing_collections:
        # vector, hybrid, rerank, and agentic all read the same default
        # collection — if it exists, all four are usable.
        available.extend(["vector", "hybrid", "rerank", "agentic"])
    if settings.contextual_collection_name in existing_collections:
        available.append("contextual")
    if Path(settings.graph_store_path).exists():
        available.append("graphrag")
    return available
