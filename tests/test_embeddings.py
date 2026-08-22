"""Sanity checks that our embedding model actually captures meaning:
related sentences should score higher on cosine similarity than
unrelated ones, even when they share few or no exact words.
"""

import pytest

from document_reviewer.embeddings import cosine_similarity, get_embeddings

VACATION_QUESTION = "How many vacation days do employees get per year?"
VACATION_ANSWER = "Employees accrue 15 days of paid time off annually."
UNRELATED = "The Widget Pro has 18 hours of battery life."


@pytest.fixture(scope="module")
def vectors():
    embedder = get_embeddings()
    sentences = [VACATION_QUESTION, VACATION_ANSWER, UNRELATED]
    return dict(zip(sentences, embedder.embed_documents(sentences)))


def test_embedding_has_expected_dimensions(vectors):
    assert len(vectors[VACATION_QUESTION]) == 384


def test_related_sentences_score_higher_than_unrelated(vectors):
    related_score = cosine_similarity(vectors[VACATION_QUESTION], vectors[VACATION_ANSWER])
    unrelated_score = cosine_similarity(vectors[VACATION_QUESTION], vectors[UNRELATED])

    assert related_score > unrelated_score
    assert related_score > 0.5
    assert unrelated_score < 0.3


def test_cosine_similarity_of_identical_vector_is_one():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)
