"""Sanity checks that the vector store retrieves the right document for
a question by meaning, not just exact word overlap — and correctly
ignores the unrelated document.
"""

from pathlib import Path

import pytest

from document_reviewer.loader import load_documents
from document_reviewer.splitter import split_documents
from document_reviewer.vector_store import build_vector_store


@pytest.fixture(scope="module")
def store():
    docs = load_documents("data")
    chunks = split_documents(docs)
    return build_vector_store(chunks)


def top_source_filename(store, question: str) -> str:
    results = store.similarity_search(question, k=1)
    return Path(results[0].metadata["source"]).name


def test_vacation_question_retrieves_vacation_policy(store):
    assert top_source_filename(store, "How many vacation days do I get?") == "vacation_policy.txt"


def test_product_question_retrieves_product_faq(store):
    assert top_source_filename(store, "Is the Widget Pro waterproof?") == "product_faq.md"
