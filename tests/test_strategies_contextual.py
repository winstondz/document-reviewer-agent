"""Tests for the contextual retrieval strategy (LLM-annotated chunks)."""

from pathlib import Path

import pytest

from document_reviewer.loader import load_documents
from document_reviewer.rag_chain import build_rag_chain
from document_reviewer.splitter import split_documents
from document_reviewer.tools.contextual import build_vector_store_contextual, contextualize_chunks


@pytest.fixture(scope="module")
def docs_and_chunks():
    docs = load_documents("data")
    chunks = split_documents(docs)
    return docs, chunks


def test_contextualize_chunks_prepends_context(docs_and_chunks):
    docs, chunks = docs_and_chunks
    enriched = contextualize_chunks(docs, chunks)

    assert len(enriched) == len(chunks)
    for original, contextualized in zip(chunks, enriched):
        assert len(contextualized.page_content) > len(original.page_content)
        assert original.page_content in contextualized.page_content


@pytest.fixture(scope="module")
def chain(docs_and_chunks):
    docs, chunks = docs_and_chunks
    build_vector_store_contextual(docs, chunks)  # fresh, deduplicated collection on disk
    return build_rag_chain(strategy="contextual")


def test_answers_vacation_question_with_correct_citation(chain):
    result = chain.invoke("How many vacation days do I get?")

    assert "15" in result["answer"]
    sources = {Path(d.metadata["source"]).name for d in result["source_documents"]}
    assert "vacation_policy.txt" in sources


def test_refuses_to_answer_unrelated_question(chain):
    result = chain.invoke("Does the Widget Pro support Bluetooth?")

    content_lower = result["answer"].lower()
    assert any(
        phrase in content_lower
        for phrase in ["don't know", "doesn't", "not mention", "not contain", "does not include", "cannot answer", "can't answer"]
    )
