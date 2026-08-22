"""Tests for the graphrag retrieval strategy (entity/relation graph)."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from document_reviewer.loader import load_documents
from document_reviewer.rag_chain import build_rag_chain
from document_reviewer.splitter import split_documents
from document_reviewer.tools.graphrag import MAX_GRAPHRAG_CHUNKS, build_graph, build_vector_store_graphrag


def test_build_graph_raises_before_exceeding_limit():
    # No real LLM calls needed here — the limit check must happen
    # before any chain.invoke.
    dummy_chunks = [Document(page_content=f"chunk {i}", metadata={"source": "dummy.txt"}) for i in range(MAX_GRAPHRAG_CHUNKS + 1)]

    with pytest.raises(ValueError):
        build_graph(dummy_chunks)


@pytest.fixture(scope="module")
def docs_and_chunks():
    docs = load_documents("data")
    chunks = split_documents(docs)
    return docs, chunks


def test_build_graph_extracts_entities_and_tracks_tokens(docs_and_chunks):
    _, chunks = docs_and_chunks
    graph = build_graph(chunks)  # well under the 50-chunk limit

    assert graph.number_of_nodes() >= 1
    nodes_lower = {str(node).lower() for node in graph.nodes}
    assert any("vacation" in node or "widget" in node for node in nodes_lower)

    assert graph.graph["total_input_tokens"] > 0
    assert graph.graph["total_output_tokens"] > 0


@pytest.fixture(scope="module")
def chain(docs_and_chunks):
    docs, chunks = docs_and_chunks
    build_vector_store_graphrag(docs, chunks)  # persists the graph + embeds chunks
    return build_rag_chain(strategy="graphrag")


def test_answers_vacation_question_with_correct_citation(chain):
    result = chain.invoke("How many vacation days do I get?")

    assert "15" in result["answer"]
    sources = {Path(d.metadata["source"]).name for d in result["source_documents"]}
    assert "vacation_policy.txt" in sources
