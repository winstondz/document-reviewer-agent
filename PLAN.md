# Document Reviewer Agent — Retrieval Strategies, Router, Observability & Web UI Plan

## Purpose

This document is an implementation plan to be executed by a coding agent. It extends the existing `document-reviewer-agent` project (a LangChain + Claude RAG CLI) in four phases:

- **Phase 1**: six interchangeable retrieval strategies, selectable via a config setting and a CLI flag.
- **Phase 2**: a router that picks a strategy automatically — heuristic (rule-based) now, with the architecture left open for an LLM-decided router later (not implemented in this pass).
- **Phase 3**: LangSmith tracing (config only, no new code, since the pipeline already uses LCEL).
- **Phase 4**: a FastAPI backend + Next.js frontend exposing manual mode (user picks a strategy) and automated mode (router picks), with every answer reporting tokens/model/latency, plus a caching layer for repeated/similar questions.

Phases are sequential and each depends on the previous one's interfaces (the router in Phase 2 selects among Phase 1's strategies; the Web UI in Phase 4 wraps Phase 2's router and Phase 1's strategies without reimplementing either). See "Build order" near the end for the full step list across all four phases.

**Existing project state this plan builds on** (do not re-implement — read these files first):
- `document_reviewer/config/settings.py` — `pydantic-settings` typed config
- `document_reviewer/config/prompts.py` — `RAG_PROMPT` (`ChatPromptTemplate`, slots `{context}` + `{question}`)
- `document_reviewer/loader.py` — `load_documents(directory) -> list[Document]`
- `document_reviewer/splitter.py` — `split_documents(docs) -> list[Document]`, `CHUNK_SIZE=500`, `CHUNK_OVERLAP=75`
- `document_reviewer/embeddings.py` — `get_embeddings() -> HuggingFaceEmbeddings`, model `sentence-transformers/all-MiniLM-L6-v2`
- `document_reviewer/vector_store.py` — `build_vector_store(chunks) -> Chroma`, `get_vector_store() -> Chroma` (Chroma, persisted at `settings.chroma_persist_dir`, collection `settings.chroma_collection_name`). **Note**: `build_vector_store` already deletes the existing collection before rebuilding (fixed bug — do not regress this).
- `document_reviewer/llm.py` — `get_llm() -> ChatAnthropic`, model `settings.chat_model` (default `claude-sonnet-5`)
- `document_reviewer/rag_chain.py` — `build_rag_chain()` returns an LCEL runnable: input a question string, output `{"answer": str, "source_documents": list[Document]}`. Uses `RETRIEVE_TOP_K = 4` and `format_docs()`.
- `app.py` — CLI with `ingest [directory]` and `ask` subcommands (argparse)
- `tests/` — pytest suite, 9 tests, real API calls (no mocking) against the 2 sample docs in `data/` (`vacation_policy.txt`, `product_faq.md`, 8 chunks total). **Note**: originally `docs/` — renamed to `data/` since `docs/` is conventionally reserved for repo documentation, not ingestion input; `docs/` now holds the retrieval-strategies writeup instead.

**Sample docs available for testing** (already exist, do not modify unless a strategy specifically requires more content):
- `data/vacation_policy.txt`
- `data/product_faq.md`

---

## Strategy Summary

| # | Strategy | Mechanism | Ingestion changes? | Relative cost |
|---|---|---|---|---|
| 1 | `plain` | Existing vector similarity search (baseline) | No | Lowest |
| 2 | `hybrid` | BM25 keyword search + vector search, combined | No | Low |
| 3 | `rerank` | Retrieve broad (top-N), rerank with a cross-encoder, keep top-k | No | Low-medium |
| 4 | `contextual` | Prepend an LLM-generated context blurb to each chunk before embedding | **Yes** — separate ingestion path | Medium (1 LLM call per chunk, ingest-time only) |
| 5 | `agentic` | LLM decides whether/how many times to search via a tool-calling loop | No | High (multiple LLM calls per question) — **needs limits** |
| 6 | `graphrag` | Extract entities/relations into a graph at ingest time; answer via graph traversal + LLM | **Yes** — separate ingestion path | Highest (1+ LLM call per chunk at ingest, plus query-time calls) — **needs limits** |

---

## Architecture: strategies are tools, not just a switch

**Design decision (see discussion that produced this plan): each retrieval strategy is implemented as a proper LangChain tool — a `@tool`-decorated function with a name, description, and schema — not just a plain function selected by a switch statement.** This matters for one concrete reason: it means manual dispatch (a human picks a strategy), the heuristic router (Phase 2, picks a strategy by rule), and a future LLM router (Phase 2, deferred — picks a strategy by binding all available tools and letting the model call one) all operate on the **same objects**. Nothing has to be re-wrapped when the LLM router is eventually added — that's the whole payoff.

Do not confuse this with Anthropic's "Agent Skills" concept (SKILL.md files, progressive disclosure) — that's a different mechanism for handing an agent reference instructions to read, not for executing retrieval. What we're building here are tools: code that executes and returns structured data.

### 1. New package: `document_reviewer/tools/`

```
document_reviewer/tools/
├── __init__.py       # registry: ALL_STRATEGY_NAMES, build_tool(name, vector_store) dispatch
├── plain.py
├── hybrid.py
├── rerank.py
├── contextual.py
├── agentic.py
└── graphrag.py
```

### 2. Common interface for the four "search tools" (`plain`, `hybrid`, `rerank`, `contextual`)

These four all do one thing: given a query, return relevant context + the source documents. They're genuinely tool-shaped. Each module exposes **two** functions:

```python
def search(vector_store: Chroma, query: str) -> tuple[str, list[Document]]:
    """Plain, ordinary function — does the actual retrieval work. Called
    directly for manual mode and by the heuristic router (both already
    know which strategy to use, so there's no need to go through
    tool-calling machinery just to invoke it)."""
    ...

def build_tool(vector_store: Chroma) -> BaseTool:
    """Wraps `search` as a @tool(response_format="content_and_artifact")
    closure bound to this vector_store, so it can be handed to
    llm.bind_tools([...]) later (used by agentic.py today, and by the
    future LLMRouter). response_format="content_and_artifact" is what
    lets the tool return both a string (for the LLM) and the raw
    Document list (for our citations) from one call — verify the exact
    invocation mechanics against current LangChain docs when
    implementing; if in doubt, keep `search` as the source of truth and
    have build_tool's closure simply call it and return its tuple."""
    ...
```

Example shape (`plain.py`):

```python
# document_reviewer/tools/plain.py
from langchain_core.tools import tool

from document_reviewer.rag_chain import RETRIEVE_TOP_K, format_docs

def search(vector_store, query: str) -> tuple[str, list]:
    docs = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K}).invoke(query)
    return format_docs(docs), docs

def build_tool(vector_store):
    @tool(response_format="content_and_artifact")
    def search_plain(query: str) -> tuple[str, list]:
        """Search the documents using standard semantic similarity.
        Good general-purpose default when no other strategy clearly fits."""
        return search(vector_store, query)
    return search_plain
```

`hybrid.py`, `rerank.py`, `contextual.py` follow the same two-function shape — only the body of `search` changes (see per-strategy sections below for what each does internally; that logic is unchanged from before, just moved inside `search` instead of `build_retriever`).

### 3. `agentic` and `graphrag` remain full chains, not single search tools

These two don't fit the "one query in, one context out" shape — `agentic` already runs its own internal tool-calling loop and produces a complete answer, not just retrieved context; `graphrag` traverses a graph and generates from that. They keep the shape from the original plan:

```python
def build_chain(vector_store: Chroma) -> Runnable:
    """Returns a full question -> {"answer": str, "source_documents": list[Document]}
    runnable, bypassing the standard rag_chain.py composition."""
```

Worth noting explicitly: `agentic.py`'s internal `search_documents` tool (see its section below) is already exactly the "strategy as a tool" pattern described above — it's the concrete example that inspired generalizing this to every strategy.

### 4. Registry (`document_reviewer/tools/__init__.py`)

```python
from typing import Literal

STRATEGY_NAMES = ("plain", "hybrid", "rerank", "contextual", "agentic", "graphrag")
StrategyName = Literal["plain", "hybrid", "rerank", "contextual", "agentic", "graphrag"]

SEARCH_TOOL_STRATEGIES = {"plain", "hybrid", "rerank", "contextual"}  # search()/build_tool()
CHAIN_STRATEGIES = {"agentic", "graphrag"}  # build_chain()
```

Add explicit imports (a dict mapping name -> module, or explicit if/elif) rather than dynamic `importlib` — prefer static, readable dispatch over cleverness here.

### 5. Config changes (`document_reviewer/config/settings.py`)

Add:
```python
retrieval_strategy: str = "plain"  # one of STRATEGY_NAMES
```

### 6. `rag_chain.py` changes

`build_rag_chain(strategy: str | None = None)` — if `strategy` is None, use `settings.retrieval_strategy`.

- If `strategy in SEARCH_TOOL_STRATEGIES`: call that module's `search(vector_store, query)` inside the retrieval step (replacing the current direct `retriever.invoke(...)` call), then compose with the existing `RAG_PROMPT` + `get_llm()` exactly as today — the generation half of `rag_chain.py` is unchanged. This is the entire point of the tool-shaped design: strategies 1-4 need zero changes to the generation half of the pipeline.
- If `strategy in CHAIN_STRATEGIES`: delegate entirely to that strategy module's `build_chain(vector_store)` and return it directly.

### 7. CLI changes (`app.py`)

- `ask` subcommand gets a new optional flag: `--strategy {plain,hybrid,rerank,contextual,agentic,graphrag}` (default: `settings.retrieval_strategy`).
- `ingest` subcommand gets the same `--strategy` flag, because `contextual` and `graphrag` need a different ingestion path (see below). Default ingestion (`plain`/`hybrid`/`rerank` all share the same ingested data) stays the current behavior.
- Print which strategy is active at the start of `ask`, so it's visible during manual testing.

---

## Per-strategy implementation details

### 1. `plain` (baseline — refactor only, no new behavior)

Move the existing `retriever = get_vector_store().as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})` line out of `rag_chain.py` into `tools/plain.py`, in the `search()`/`build_tool()` shape defined above (full example already given in the Architecture section — no need to repeat it here).

Verify all 9 existing tests still pass after this refactor before moving on — this step should be a pure no-op from the user's perspective.

### 2. `hybrid`

**New dependency**: `rank_bm25` (add to `requirements.txt`; pulled in automatically by `langchain_community.retrievers.BM25Retriever`, but pin it explicitly).

```python
# document_reviewer/tools/hybrid.py
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool

from document_reviewer.rag_chain import RETRIEVE_TOP_K, format_docs

def _build_ensemble_retriever(vector_store):
    # BM25Retriever needs the raw documents, not the vector store —
    # pull them back out via the underlying Chroma collection.
    all_docs = vector_store.get()  # returns dict with "documents" and "metadatas"
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = RETRIEVE_TOP_K

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],  # equal weight; make this tunable later if desired
    )

def search(vector_store, query: str) -> tuple[str, list[Document]]:
    docs = _build_ensemble_retriever(vector_store).invoke(query)
    return format_docs(docs), docs

def build_tool(vector_store):
    @tool(response_format="content_and_artifact")
    def search_hybrid(query: str) -> tuple[str, list[Document]]:
        """Search using both keyword matching and semantic similarity.
        Best when the question contains exact terms, codes, quoted
        phrases, or specific names that plain semantic search might miss."""
        return search(vector_store, query)
    return search_hybrid
```

**Test**: add `tests/test_tools_hybrid.py` — assert a keyword-heavy query (e.g. exact phrase "IP67") retrieves the right chunk at least as well as `plain`, and that it still passes the two existing correctness assertions from `test_rag_chain.py` (vacation days -> 15, cites `vacation_policy.txt`).

### 3. `rerank`

**No new dependency** — `sentence-transformers` (already installed) ships `CrossEncoder`.

```python
# document_reviewer/tools/rerank.py
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.tools import tool

from document_reviewer.rag_chain import format_docs

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local, no API key
RETRIEVE_BROAD_K = 15  # cast a wider net before reranking
RETRIEVE_FINAL_K = 4   # keep this many after reranking

def _build_reranking_retriever(vector_store):
    broad_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_BROAD_K})
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=RETRIEVE_FINAL_K)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=broad_retriever,
    )

def search(vector_store, query: str) -> tuple[str, list[Document]]:
    docs = _build_reranking_retriever(vector_store).invoke(query)
    return format_docs(docs), docs

def build_tool(vector_store):
    @tool(response_format="content_and_artifact")
    def search_rerank(query: str) -> tuple[str, list[Document]]:
        """Search broadly, then re-rank results for relevance before
        returning the best matches. Higher precision than plain search,
        useful when the top result from plain search often feels
        slightly off-target."""
        return search(vector_store, query)
    return search_rerank
```

Add `RERANK_MODEL`, `RETRIEVE_BROAD_K` constants; make them overridable via `Settings` if the agent judges it worthwhile (not required).

**Test**: `tests/test_tools_rerank.py` — same two correctness assertions as `hybrid`. Note the cross-encoder model download (~small, <100MB) on first run — allow extra timeout in test config if needed.

### 4. `contextual` (Anthropic's Contextual Retrieval technique)

This strategy changes **ingestion**, not just retrieval. It needs its own vector store collection so it doesn't collide with the default one (different embeddings content = different chunks = must not share a Chroma collection name with `plain`/`hybrid`/`rerank`).

**Config addition**: `contextual_collection_name: str = "documents_contextual"` in `Settings`.

**New ingestion function** (`document_reviewer/tools/contextual.py`):

```python
CONTEXTUAL_SYSTEM_PROMPT = """You are helping index a document for search. \
Given the full document and one chunk from it, write a short (1-2 \
sentence) context statement that situates the chunk within the \
document, to improve search retrieval of the chunk. Answer only with \
the succinct context, nothing else."""

CONTEXTUAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUAL_SYSTEM_PROMPT),
    ("human", "<document>\n{full_document}\n</document>\n\n<chunk>\n{chunk}\n</chunk>"),
])

def contextualize_chunks(docs: list[Document], chunks: list[Document]) -> list[Document]:
    """Returns new Document objects with an LLM-generated context blurb
    prepended to each chunk's page_content, before embedding.
    Makes one LLM call per chunk — see cost note below.
    """
    llm = get_llm()  # use a cheap/fast model override here if desired, e.g. claude-haiku-4-5
    chain = CONTEXTUAL_PROMPT | llm

    # Build a lookup from source filename -> full original document text,
    # so each chunk can be given its parent document for context.
    full_text_by_source = {doc.metadata["source"]: doc.page_content for doc in docs}

    contextualized = []
    for chunk in chunks:
        full_document = full_text_by_source[chunk.metadata["source"]]
        context_blurb = chain.invoke({"full_document": full_document, "chunk": chunk.page_content}).content
        new_content = f"{context_blurb}\n\n{chunk.page_content}"
        contextualized.append(Document(page_content=new_content, metadata=chunk.metadata))
    return contextualized

def build_vector_store_contextual(docs: list[Document], chunks: list[Document]) -> Chroma:
    """Ingestion entry point for the contextual strategy. Wraps
    vector_store.build_vector_store but targets the separate
    contextual collection and pre-processes chunks first."""
    enriched_chunks = contextualize_chunks(docs, chunks)
    # reuse vector_store.py's Chroma.from_documents pattern, but with
    # settings.contextual_collection_name instead of the default —
    # either parameterize build_vector_store() to accept a collection
    # name override, or duplicate the minimal logic here. Prefer
    # parameterizing vector_store.build_vector_store(chunks, collection_name=...)
    # to avoid duplicating the delete-existing-collection fix.

def search(vector_store_contextual: Chroma, query: str) -> tuple[str, list[Document]]:
    docs = vector_store_contextual.as_retriever(search_kwargs={"k": 4}).invoke(query)
    return format_docs(docs), docs

def build_tool(vector_store_contextual: Chroma):
    @tool(response_format="content_and_artifact")
    def search_contextual(query: str) -> tuple[str, list[Document]]:
        """Search an index where each chunk was pre-annotated with
        context about its place in the source document. Best for
        questions where a chunk's meaning depends on surrounding
        context the plain chunk text alone wouldn't convey."""
        return search(vector_store_contextual, query)
    return search_contextual
```

**Required refactor**: parameterize `vector_store.build_vector_store(chunks, collection_name: str | None = None)` and `get_vector_store(collection_name: str | None = None)`, defaulting to `settings.chroma_collection_name` when not given. This lets `contextual` (and `graphrag`, if it also uses Chroma for its base chunks) reuse the same duplicate-safe build logic instead of copy-pasting it.

**CLI wiring**: `python app.py ingest docs --strategy contextual` calls `contextualize_chunks` then builds into the contextual collection. `python app.py ask --strategy contextual` opens that collection.

**Cost note — put this directly in the code as a comment and print a warning to the user before running**: this makes **one LLM call per chunk** at ingest time. For our 8 sample chunks that's 8 calls (cheap, seconds). Flag in the code that this does NOT scale silently — for a real document set of e.g. 1000 chunks, ingestion would make 1000 LLM calls. Recommended real-world guardrail (implement this): a `MAX_CONTEXTUAL_CHUNKS` config value (default e.g. 200) that `contextualize_chunks` refuses to exceed without an explicit `--force` flag, printing a clear error otherwise ("would require N calls, limit is 200; pass --force to proceed").

**Test**: `tests/test_strategies_contextual.py` — run `contextualize_chunks` on our 8 sample chunks (small enough that no limiting logic engages), assert each output chunk's `page_content` is longer than the original (context was prepended) and still contains the original chunk text as a substring. Then assert the standard retrieval correctness checks against the contextual collection.

### 5. `agentic` — REQUIRES LIMITS

The LLM gets the retriever as a callable tool and decides for itself whether/how many times to call it, instead of always retrieving exactly once.

```python
# document_reviewer/tools/agentic.py
from langchain_core.tools import tool

MAX_AGENT_ITERATIONS = 3       # hard cap on tool-calling loop turns
MAX_SEARCHES_PER_QUESTION = 3  # hard cap on retriever calls within one question
AGENT_MAX_TOKENS = 1024        # cap per LLM call in the loop

def build_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    search_call_count = {"count": 0}  # closure-scoped counter, reset per invocation

    @tool
    def search_documents(query: str) -> str:
        """Search the user's documents for relevant passages. Use this
        when you need information to answer the question. You may call
        this tool multiple times with refined queries if the first
        search doesn't return what you need, up to a limit."""
        if search_call_count["count"] >= MAX_SEARCHES_PER_QUESTION:
            return "Search limit reached. Answer with what you have."
        search_call_count["count"] += 1
        docs = retriever.invoke(query)
        return format_docs(docs)  # reuse rag_chain.format_docs

    llm = get_llm()
    llm_with_tools = llm.bind_tools([search_documents])

    # Implement as an explicit bounded loop (NOT langgraph / create_react_agent,
    # to keep this dependency-light and keep the iteration cap trivially
    # auditable): call llm_with_tools, check for tool_calls, execute
    # search_documents for each, append ToolMessage results, repeat until
    # no more tool_calls OR MAX_AGENT_ITERATIONS reached. On hitting the
    # cap without a final answer, force one more call with the
    # search_documents tool removed from bind_tools so the model must
    # answer from what it has.
    #
    # Track source_documents actually returned across all search calls
    # (accumulate in a list from within the tool) so the final output
    # can still cite real filenames, matching the existing
    # {"answer": str, "source_documents": list[Document]} return shape.
```

**Explicit requirement from the user — implement all of these**:
- `MAX_AGENT_ITERATIONS` and `MAX_SEARCHES_PER_QUESTION` are hard caps, not suggestions — the loop must physically stop, not just get instructed to stop.
- Print (or log) the number of LLM calls and total tokens used per question when this strategy runs — pull `response.usage_metadata` (available on `ChatAnthropic` responses: `input_tokens`, `output_tokens`) and sum across every call in the loop.
- `AGENT_MAX_TOKENS` bounds each individual call's output size.

**Test**: `tests/test_strategies_agentic.py` — assert the loop terminates (doesn't hang) on both a simple question (should resolve in 1 search) and a deliberately vague/multi-part question (may use multiple searches, but must stop at `MAX_SEARCHES_PER_QUESTION`). Assert token usage is captured and printed/returned. Given real API calls, keep this test's question set small (2 cases max) to control CI cost.

### 6. `graphrag` — REQUIRES LIMITS, TREAT AS OPT-IN/STRETCH

Highest cost and complexity of the six. Recommend implementing this last, and gating it behind an explicit confirmation since a careless run on a large doc set is the easiest way to accidentally burn a lot of tokens.

**New dependency**: `networkx` (in-memory graph, no external graph database needed at this scale — add to `requirements.txt`).

```python
# document_reviewer/tools/graphrag.py
import networkx as nx

MAX_GRAPHRAG_CHUNKS = 50          # hard cap on chunks processed for extraction
EXTRACTION_MAX_TOKENS = 512       # cap per extraction LLM call
MAX_ENTITIES_PER_CHUNK = 10       # sanity cap on extraction output size

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract entities and relationships from the given text \
chunk. Respond as JSON: {{"entities": ["..."], "relations": \
[{{"source": "...", "relation": "...", "target": "..."}}]}}. Extract at \
most {max_entities} entities. If none found, return empty lists."""),
    ("human", "{chunk}"),
])

def build_graph(chunks: list[Document]) -> nx.MultiDiGraph:
    if len(chunks) > MAX_GRAPHRAG_CHUNKS:
        raise ValueError(
            f"GraphRAG ingestion would process {len(chunks)} chunks, "
            f"exceeding the safety limit of {MAX_GRAPHRAG_CHUNKS}. "
            "Reduce the document set or raise MAX_GRAPHRAG_CHUNKS deliberately."
        )

    llm = get_llm()
    chain = EXTRACTION_PROMPT | llm  # consider output_config.format for
                                       # guaranteed-valid JSON — see
                                       # shared structured-outputs pattern
                                       # already usable via ChatAnthropic
    graph = nx.MultiDiGraph()
    total_input_tokens = total_output_tokens = 0

    for chunk in chunks:
        response = chain.invoke({
            "chunk": chunk.page_content,
            "max_entities": MAX_ENTITIES_PER_CHUNK,
        })
        total_input_tokens += response.usage_metadata["input_tokens"]
        total_output_tokens += response.usage_metadata["output_tokens"]
        data = json.loads(response.content)  # parse; handle malformed JSON defensively
        for entity in data.get("entities", []):
            graph.add_node(entity, source=chunk.metadata["source"])
        for rel in data.get("relations", []):
            graph.add_edge(rel["source"], rel["target"], relation=rel["relation"], source_doc=chunk.metadata["source"])

    print(f"GraphRAG extraction: {len(chunks)} chunks, "
          f"{total_input_tokens} input / {total_output_tokens} output tokens")
    return graph
```

Query-time (`build_chain`) should: extract likely entities from the question (reuse a lightweight version of the same extraction call, or simple keyword match against known graph node names to avoid another LLM call), traverse the graph for connected nodes/edges, pull the associated chunks back from the base vector store by source, and feed those into the same `RAG_PROMPT` + `get_llm()` pattern as everything else — keeping the final answer generation step consistent across all six strategies.

**Explicit requirement from the user — implement all of these**:
- `MAX_GRAPHRAG_CHUNKS` is a hard pre-flight check that **raises before making any LLM calls** if exceeded — never silently truncate and proceed, since that produces an incomplete graph without warning.
- Print running token totals (input/output) as extraction proceeds, not just a final summary — for a slow operation like this, the user should see progress.
- `EXTRACTION_MAX_TOKENS` and `MAX_ENTITIES_PER_CHUNK` bound both cost and pathological LLM outputs (e.g. a chunk that causes the model to list hundreds of spurious entities).

**Test**: `tests/test_strategies_graphrag.py` — build a graph from the 8 sample chunks (well under the 50-chunk limit), assert the graph has at least 1 node and the vacation-policy-related nodes exist, assert total token counts are captured and non-zero. Separately, add a test that `build_graph` raises `ValueError` when given a fake list of 51+ dummy chunks (no real LLM calls needed for this one — the limit check must happen before any `chain.invoke`).

---

## PHASE 2 — Router (heuristic now, LLM-decided later)

Instead of the user always passing `--strategy`, add an **automated mode** where a router picks the strategy for you. Build this so a smarter router can be dropped in later without touching call sites.

### Interface (design for extension from day one)

```python
# document_reviewer/router.py
from typing import Protocol

class Router(Protocol):
    def select_strategy(self, question: str, available_strategies: list[str]) -> str:
        """Return the name of the strategy to use for this question.
        Must only return a name present in available_strategies (i.e.
        one whose index actually exists — see constraint below)."""
        ...
```

`available_strategies` is passed in by the caller (`app.py` / the API layer, once it exists) as whichever indices have actually been ingested — the router must never select a strategy whose collection/graph doesn't exist on disk. This constraint doesn't change later when the LLM router is added; enforce it in one place (the caller), not duplicated per-router.

### `HeuristicRouter` (implement now)

Simple rule-based dispatch, no LLM call — cheap and deterministic:

```python
# document_reviewer/router.py (cont.)
import re

class HeuristicRouter:
    def select_strategy(self, question: str, available_strategies: list[str]) -> str:
        def available(name: str) -> bool:
            return name in available_strategies

        # Quoted phrase or code-like token -> keyword search matters
        if available("hybrid") and (re.search(r'"[^"]+"', question) or re.search(r"\b[A-Z]{2,}\d+\b", question)):
            return "hybrid"

        # Multiple entities / "and" / "compare" / "relationship" -> graph
        if available("graphrag") and re.search(r"\b(compare|relationship|between|and how)\b", question, re.I):
            return "graphrag"

        # Long or multi-part question -> let the agent decide how much to search
        if available("agentic") and (len(question.split()) > 20 or "?" in question[:-1]):
            return "agentic"

        # Default: best general-purpose option that exists
        for fallback in ("rerank", "hybrid", "plain"):
            if available(fallback):
                return fallback
        return "plain"
```

These specific rules are a starting point, not gospel — reasonable to adjust once you see real question patterns. What matters structurally: **the `Router` protocol and the `available_strategies` constraint stay fixed** so `LLMRouter` (later) is a drop-in replacement.

### `LLMRouter` (do NOT implement yet — stub only)

Because every search strategy is already a tool (see Architecture section above), this becomes small once it's time to build it: bind the `build_tool(vector_store)` objects for whichever of `plain`/`hybrid`/`rerank`/`contextual` currently have an index (`llm.bind_tools([...])`), and let the model's native tool-calling pick one. `agentic` and `graphrag` stay whole-chain choices offered separately (they're full answer-producing chains, not single search tools — see Architecture section 3), so the LLM router only ever chooses among the four search tools; a heuristic rule (or a second, cheap decision) still picks between "use the search-tool router" vs "use agentic" vs "use graphrag".

```python
class LLMRouter:
    """Not implemented in this pass. Will bind the four search-tool
    strategies' build_tool(vector_store) objects — whichever have an
    index available — via llm.bind_tools([...]) and let the model pick
    by calling one. Must still respect available_strategies (only bind
    tools for strategies with an existing index). Add later, once
    HeuristicRouter is in place and its rules have been validated
    against real usage — see PLAN.md discussion for why this is
    deferred despite being cheap to build."""
    def select_strategy(self, question: str, available_strategies: list[str]) -> str:
        raise NotImplementedError("Phase 2.5 — not part of this plan")
```

### Config

Add `router_type: str = "heuristic"` to `Settings` (values: `"heuristic"` now, `"llm"` reserved for later — validate against this set even though only one is implemented, so the config surface doesn't need to change again).

### CLI wiring

`python app.py ask --mode automated` uses the router (reading whichever strategy indices exist); `python app.py ask --strategy hybrid` (existing flag, unchanged) forces manual override. `--mode automated` and `--strategy X` are mutually exclusive — validate and error clearly if both are passed.

**Test**: `tests/test_router.py` — unit tests only, no LLM/API calls needed since `HeuristicRouter` is pure logic. Assert each rule branch picks the expected strategy given canned `available_strategies` lists, including the fallback chain when a preferred strategy isn't available.

---

## PHASE 3 — Observability: LangSmith

LangChain has built-in LangSmith tracing — this is config, not new code, for anything already built with LCEL (every chain in this project already is).

### Setup

Add to `.env.example`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=document-reviewer-agent
```

Add corresponding **optional** fields to `Settings` (`langchain_tracing_v2: bool = False`, `langchain_api_key: str = ""`, `langchain_project: str = "document-reviewer-agent"`) purely so they're validated/typed like everything else — LangChain itself reads these as plain environment variables, so `Settings` doesn't need to pass them anywhere explicitly as long as `.env` is loaded before any chain runs (already true today via `load_dotenv`-equivalent in `pydantic-settings`).

No code changes needed in `rag_chain.py`, `tools/`, or `router.py` — every `.invoke()` call on an LCEL runnable is traced automatically once the env vars are set. This is why LCEL was worth using from the start instead of hand-rolled orchestration.

**Verify**: run `python app.py ask` with tracing env vars set, ask one question, confirm a trace appears in the LangSmith project — this is a manual check, not a pytest test (no assertion to make against a third-party dashboard).

---

## PHASE 4 — Web UI: FastAPI backend + Next.js frontend

Two modes in the UI, mapping directly onto Phase 2's router:
- **Manual**: user picks a strategy from a dropdown -> API called with `strategy=<name>`.
- **Automated**: no strategy chosen -> API uses the router (`Router.select_strategy`).

### Backend: FastAPI (new top-level directory `api/`)

Wraps the existing `document_reviewer` package — does not reimplement any RAG logic, only exposes it over HTTP.

```
api/
├── main.py           # FastAPI app, route definitions
└── schemas.py        # Pydantic request/response models
```

**Endpoints**:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Multipart file upload(s) -> save to a temp/upload dir -> run `load_documents` + `split_documents` + `build_vector_store` (per selected strategy's ingestion path) |
| `POST` | `/ask` | `{"question": str, "mode": "manual" \| "automated", "strategy": str \| None}` -> runs the chain, returns the response schema below |
| `GET` | `/strategies` | Returns which strategies currently have a built index available (drives the manual-mode dropdown and the automated-mode router's `available_strategies`) |

**Response schema for `/ask`** (this is the "print tokens/model/latency" requirement):

```python
# api/schemas.py
from pydantic import BaseModel
from document_reviewer.config.prompts import ...  # no import needed here, illustrative only

class AskResponse(BaseModel):
    answer: str
    sources: list[str]          # filenames, e.g. ["vacation_policy.txt"]
    strategy_used: str          # which strategy actually ran (relevant in automated mode)
    model: str                  # settings.chat_model
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool                # see Phase 4 caching below
```

Populate `input_tokens`/`output_tokens` from `response.usage_metadata` on the final LLM call (same field already used in the `agentic`/`graphrag` token tracking above — reuse that pattern here rather than inventing a second way to read usage). Populate `latency_ms` by wrapping the chain's `.invoke()` call with `time.perf_counter()` before/after — measure the whole chain invocation, not just the final LLM call, so retrieval time is included.

**Reused, not duplicated**: `/ingest` and `/ask` call directly into `document_reviewer/` functions already built in Phase 1 (`build_rag_chain`, `tools.*`, `router.HeuristicRouter`) — the API layer is a thin adapter, not a second implementation.

### Frontend: Next.js (new top-level directory `web/`)

Minimal pages, no need for anything elaborate:
- **Upload page**: file picker -> `POST /ingest`.
- **Ask page**: mode toggle (Manual / Automated); when Manual, a strategy dropdown populated from `GET /strategies`; a question input; on submit, `POST /ask` and render `answer`, `sources`, and a small metadata line showing `strategy_used`, `model`, `input_tokens + output_tokens`, `latency_ms`, and a "(cached)" badge when `cached: true`.

No auth, no persistence beyond what the backend already does, no styling requirements beyond basic usability — this is a working demo UI, not a polished product.

---

## PHASE 4 (cont.) — Caching for same/similar questions

Two layers, build the first before considering the second:

### 1. Exact-match cache (do this)

Simplest possible: a dict (or SQLite table) keyed by `(question_normalized, strategy_used)` -> the full `AskResponse` (minus recomputing `latency_ms`, which should reflect the *original* run so cached responses can honestly report a near-zero cache-hit latency separately, or just show `cached: true` and omit a misleading latency figure — prefer the latter for simplicity). `question_normalized` = lowercased + whitespace-collapsed question string.

Reuse the existing `chroma_db` directory pattern by putting this in a new SQLite file (e.g. `cache.db`) rather than inventing a new persistence mechanism — `sqlite3` is stdlib, no new dependency.

```python
# document_reviewer/cache.py
import sqlite3

def get_cached_answer(question_normalized: str, strategy: str) -> AskResponse | None: ...
def store_cached_answer(question_normalized: str, strategy: str, response: AskResponse) -> None: ...
```

Wire into `/ask`: check cache first (keyed on the *actual* strategy used — in automated mode, resolve the router's choice before checking cache, since the same question might route differently if `available_strategies` changes between calls), return immediately with `cached: true` on hit; otherwise run the chain, store, return with `cached: false`.

### 2. Similar-question cache (stretch — only after #1 works)

Embed each incoming question (reuse `get_embeddings()` — same model already loaded for retrieval, no new model needed), compare against embeddings of previously cached questions via cosine similarity (reuse `embeddings.cosine_similarity` — already written and tested in Phase 1's predecessor work), and treat anything above a threshold (start at `0.95`, tune later) as a cache hit. Store question embeddings alongside the exact-match cache table rather than as a separate system.

**Explicit limit, matching the pattern from `agentic`/`graphrag`**: cap the number of cached-question embeddings compared against per lookup (e.g. only check the most recent 500 cached questions) so cache lookup time doesn't grow unbounded as the cache fills.

**Test**: `tests/test_cache.py` — unit tests against `document_reviewer/cache.py` directly (no API/LLM calls needed): store then retrieve exact match; confirm a miss returns `None`; for the similarity layer, assert a paraphrase of a cached question above the threshold hits and a clearly unrelated question misses (can use the same fixture sentences from `test_embeddings.py`).

---

## Build order (do in this sequence)

**Phase 1 — Retrieval strategies** (all detail above this point in the doc):
1. Scaffold the switch mechanism with only `plain` wired through (registry, config field, CLI flag, `rag_chain.py` dispatch). Run the full existing test suite — must still pass unchanged.
2. `hybrid` — add dependency, implement, test.
3. `rerank` — implement, test.
4. `contextual` — refactor `vector_store.py` to accept a `collection_name` param first (needed by this and `graphrag`), then implement ingestion + retrieval, test.
5. `agentic` — implement bounded loop with token tracking, test with a small question set.
6. `graphrag` — implement last, with the pre-flight chunk-count guard as the very first thing written.
7. Update `requirements.txt` with all new dependencies (`rank_bm25`, `networkx`).

**Phase 2 — Router**:
8. Implement `Router` protocol + `HeuristicRouter` in `document_reviewer/router.py`, `router_type` config, `--mode automated` CLI flag. Unit test, no API calls.

**Phase 3 — Observability**:
9. Add LangSmith env vars to `.env.example` and `Settings`. Manually verify traces appear.

**Phase 4 — Web UI**:
10. Build `api/` (FastAPI) — thin wrapper over existing `document_reviewer` functions, including the `AskResponse` metadata fields and the exact-match cache.
11. Build `web/` (Next.js) — upload page, ask page with manual/automated toggle and metadata display.
12. (Stretch) Add similarity-based cache layer on top of the exact-match cache.
13. Update `README.md` documenting the full system: CLI usage, `--strategy`/`--mode` flags, how to run the API + web UI, and the cost/limit notes for `agentic`/`graphrag`/`contextual`.

## Non-goals (explicitly out of scope for this plan)

- Do not add a hosted reranker (Voyage/Cohere) — stay local/free, consistent with the embeddings decision made earlier in this project.
- Do not add a real graph database (Neo4j etc.) — `networkx` in-memory is sufficient at this project's scale.
- Do not implement `LLMRouter` in this pass — stub only, per Phase 2. Do not remove the `Router` protocol boundary in order to "simplify" — that boundary is what makes the later LLM router a drop-in change.
- Do not add authentication, multi-user support, or persistent chat history to the Web UI — single-user local demo only.
- Do not remove or change the existing `plain` strategy behavior beyond the file-move refactor in Phase 1 step 1 — it must remain the default and behave identically to today.
