"""End-to-end test of the full RAG pipeline: real retrieval + real LLM
call. This is what caught the duplicate-chunk bug in vector_store.py —
build_vector_store wasn't clearing the existing collection, so repeated
ingests piled up duplicates that crowded the actual answer out of the
top-k results. Rebuilding the store fresh before each test run guards
against that regressing.
"""

from pathlib import Path

import pytest

from document_reviewer.loader import load_documents
from document_reviewer.rag_chain import build_rag_chain
from document_reviewer.splitter import split_documents
from document_reviewer.vector_store import build_vector_store


@pytest.fixture(scope="module")
def chain():
    docs = load_documents("data")
    chunks = split_documents(docs)
    build_vector_store(chunks)  # fresh, deduplicated collection on disk
    return build_rag_chain()


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
