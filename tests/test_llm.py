"""Confirms the RAG prompt actually does its job: Claude answers from
the given context when the answer is there, and admits it doesn't know
when it isn't — rather than falling back to outside knowledge. This is
step 6 in isolation: context is supplied by hand here, not retrieved
yet (that's rag_chain.py, next).

Makes real API calls, so keep this file small.
"""

from document_reviewer.config.prompts import RAG_PROMPT
from document_reviewer.llm import get_llm

VACATION_CONTEXT = (
    "Source: vacation_policy.txt\n"
    "All full-time employees accrue 15 days of paid vacation per calendar year."
)


def test_answers_from_provided_context():
    llm = get_llm()
    chain = RAG_PROMPT | llm
    response = chain.invoke(
        {"context": VACATION_CONTEXT, "question": "How many vacation days do employees get?"}
    )
    assert "15" in response.content


def test_admits_when_context_lacks_the_answer():
    llm = get_llm()
    chain = RAG_PROMPT | llm
    response = chain.invoke(
        {"context": VACATION_CONTEXT, "question": "What is the battery life of the Widget Pro?"}
    )
    content_lower = response.content.lower()
    assert any(
        phrase in content_lower
        for phrase in ["don't know", "doesn't know", "no information", "not contain", "does not include", "can't answer", "cannot answer"]
    )
    # And it should not fabricate a real-looking battery life number
    assert "hours" not in content_lower or "battery" not in content_lower
