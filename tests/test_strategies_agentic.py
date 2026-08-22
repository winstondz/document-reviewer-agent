"""Tests for the agentic retrieval strategy (bounded tool-calling loop).

Real API calls, so keep the question set small (2 cases) to control cost.
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
    return build_rag_chain(strategy="agentic")


def test_simple_question_terminates_with_correct_citation(chain):
    result = chain.invoke("How many vacation days do I get?")

    assert "15" in result["answer"]
    sources = {Path(d.metadata["source"]).name for d in result["source_documents"]}
    assert "vacation_policy.txt" in sources


def test_vague_multipart_question_terminates(chain):
    # Deliberately broad/multi-part — may prompt more than one search
    # call, but the loop must still terminate within the hard caps
    # rather than hang or loop indefinitely.
    result = chain.invoke(
        "What are the vacation policy rules and is the Widget Pro waterproof and what is its warranty?"
    )

    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
