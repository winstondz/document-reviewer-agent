"""GraphRAG: extract entities and relationships from chunks into a
graph at ingest time; answer questions by traversing the graph for
connected facts instead of doing chunk similarity search.

Highest cost and complexity of the six strategies — one LLM call per
chunk at ingest (like contextual), plus this is the only strategy that
answers via graph traversal rather than a nearest-neighbor search.
Treat as opt-in/stretch: gated behind --force at the CLI (see app.py)
since a careless run on a large document set is the easiest way to
accidentally burn a lot of tokens.

Like agentic, this doesn't fit the search()/build_tool() shape (see
tools/__init__.py docstring) — it exposes build_chain(vector_store).
Unlike contextual, it doesn't need its own vector store collection:
the graph is a separate on-disk structure (pickled to
settings.graph_store_path), and query time still pulls raw chunk text
back out of the *default* Chroma collection by source filename, so
ingestion also embeds the chunks there as normal.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from document_reviewer.config import settings
from document_reviewer.config.prompts import RAG_PROMPT
from document_reviewer.llm import get_llm
from document_reviewer.rag_chain import extract_text_content, format_docs

# Hard caps — MAX_GRAPHRAG_CHUNKS is checked BEFORE any LLM call, so a
# large document set fails fast with a clear error instead of silently
# producing an incomplete graph partway through.
MAX_GRAPHRAG_CHUNKS = 50
EXTRACTION_MAX_TOKENS = 512
MAX_ENTITIES_PER_CHUNK = 10

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract entities and relationships from the given text \
chunk. Respond as JSON: {{"entities": ["..."], "relations": \
[{{"source": "...", "relation": "...", "target": "..."}}]}}. Extract at \
most {max_entities} entities. If none found, return empty lists. \
Respond with ONLY the JSON object, no other text.""",
        ),
        ("human", "{chunk}"),
    ]
)


def _parse_extraction(content) -> dict:
    """Parse the extraction LLM's JSON response defensively — malformed
    or non-JSON output (the model ignoring instructions, a truncated
    response, etc.) degrades to "no entities found" for that chunk
    rather than crashing the whole ingestion run."""
    text = extract_text_content(content).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}


def build_graph(chunks: list[Document]) -> nx.MultiDiGraph:
    """Extract entities/relations from `chunks` into a graph. Makes one
    LLM call per chunk — see the module docstring's cost note.

    Token totals are stored as graph-level attributes (graph.graph[...])
    rather than returned separately, so the graph stays the single
    source of truth for what this run cost.
    """
    if len(chunks) > MAX_GRAPHRAG_CHUNKS:
        raise ValueError(
            f"GraphRAG ingestion would process {len(chunks)} chunks, "
            f"exceeding the safety limit of {MAX_GRAPHRAG_CHUNKS}. "
            "Reduce the document set or raise MAX_GRAPHRAG_CHUNKS deliberately."
        )

    llm = get_llm().bind(max_tokens=EXTRACTION_MAX_TOKENS)
    chain = EXTRACTION_PROMPT | llm

    graph = nx.MultiDiGraph()
    total_input_tokens = 0
    total_output_tokens = 0

    for i, chunk in enumerate(chunks):
        response = chain.invoke({"chunk": chunk.page_content, "max_entities": MAX_ENTITIES_PER_CHUNK})
        usage = response.usage_metadata or {}
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)

        data = _parse_extraction(response.content)
        source = chunk.metadata.get("source", "unknown")

        for entity in data.get("entities", [])[:MAX_ENTITIES_PER_CHUNK]:
            graph.add_node(entity, source=source)
        for rel in data.get("relations", []):
            if "source" in rel and "target" in rel:
                graph.add_edge(rel["source"], rel["target"], relation=rel.get("relation", ""), source_doc=source)

        print(
            f"GraphRAG extraction: chunk {i + 1}/{len(chunks)} — "
            f"running totals: {total_input_tokens} input / {total_output_tokens} output tokens"
        )

    graph.graph["total_input_tokens"] = total_input_tokens
    graph.graph["total_output_tokens"] = total_output_tokens
    print(
        f"GraphRAG extraction complete: {len(chunks)} chunk(s), "
        f"{total_input_tokens} input / {total_output_tokens} output tokens, "
        f"{graph.number_of_nodes()} entities, {graph.number_of_edges()} relations"
    )
    return graph


def save_graph(graph: nx.MultiDiGraph, path: str | None = None) -> None:
    with open(path or settings.graph_store_path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path: str | None = None) -> nx.MultiDiGraph:
    path = path or settings.graph_store_path
    if not Path(path).exists():
        raise FileNotFoundError(f"No graph found at '{path}'. Run `python app.py ingest --strategy graphrag --force` first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_vector_store_graphrag(docs: list[Document], chunks: list[Document]):
    """Ingestion entry point for the graphrag strategy: extract and
    persist the graph, and embed the chunks into the default Chroma
    collection so query time can pull raw chunk text back out by
    source filename (see module docstring)."""
    from document_reviewer.vector_store import build_vector_store

    graph = build_graph(chunks)
    save_graph(graph)
    build_vector_store(chunks)
    return graph


def _find_matching_entities(graph: nx.MultiDiGraph, question: str) -> list[str]:
    """Simple keyword match against known entity names, rather than a
    second LLM call, to identify which graph nodes the question is
    likely asking about."""
    question_lower = question.lower()
    return [node for node in graph.nodes if str(node).lower() in question_lower]


def _traverse(graph: nx.MultiDiGraph, entities: list[str]) -> set[str]:
    """One hop out from each matched entity, collecting the source
    files of everything reachable, so query time can pull back only
    the chunks actually connected to what the question is asking."""
    sources: set[str] = set()
    for entity in entities:
        if entity not in graph:
            continue
        sources.add(graph.nodes[entity].get("source", ""))
        for neighbor in list(graph.successors(entity)) + list(graph.predecessors(entity)):
            if neighbor in graph:
                sources.add(graph.nodes[neighbor].get("source", ""))
    sources.discard("")
    return sources


def _fetch_chunks_by_source(vector_store, sources: set[str]) -> list[Document]:
    all_docs = vector_store.get()
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        if meta.get("source") in sources
    ]


def build_chain(vector_store):
    graph = load_graph()
    generate = RAG_PROMPT | get_llm() | StrOutputParser()

    def _run(question: str) -> dict:
        entities = _find_matching_entities(graph, question)
        sources = _traverse(graph, entities)
        source_documents = _fetch_chunks_by_source(vector_store, sources)

        if not source_documents:
            # No graph entities matched the question — fall back to a
            # plain vector search over the same collection rather than
            # answering with no context at all.
            source_documents = vector_store.as_retriever(search_kwargs={"k": 4}).invoke(question)

        answer = generate.invoke({"context": format_docs(source_documents), "question": question})
        return {"answer": answer, "source_documents": source_documents}

    return RunnableLambda(_run)
