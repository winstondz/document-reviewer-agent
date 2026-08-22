"""The full RAG pipeline: question -> retrieve relevant chunks -> stuff
them into the prompt -> ask Claude -> return the answer plus which
source documents it was grounded in.

This is where every earlier step connects:
  vector_store.get_vector_store()  (step 5) -> retriever
  config/prompts.py RAG_PROMPT     (step 6) -> {context} + {question}
  llm.get_llm()                    (step 6) -> Claude
LCEL (the `|` pipe operator) wires them into one runnable pipeline.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from document_reviewer.config.prompts import RAG_PROMPT
from document_reviewer.config.settings import settings
from document_reviewer.llm import get_llm

RETRIEVE_TOP_K = 4


class RagResult(TypedDict):
    answer: str
    source_documents: list[Document]


def format_docs(docs: list[Document]) -> str:
    """Turn retrieved chunks into the {context} string the prompt sees,
    labeling each chunk with its source file so Claude can cite it.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def extract_text_content(content) -> str:
    """AIMessage.content is a plain str for a simple text reply, but a
    list of content blocks (thinking/text/tool_use) once tools are
    bound — extract just the text either way. Used by any strategy
    that binds tools (agentic) or needs to parse text out of a response
    that might come back as blocks (graphrag's JSON extraction calls).
    """
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")


def build_rag_chain(strategy: str | None = None):
    """Build the full pipeline as a single LCEL runnable.

    Input:  a question string.
    Output: a dict with "answer" (str) and "source_documents" (the raw
            Document chunks that were retrieved) — kept separate from
            the answer text so the caller can display real filenames
            even if Claude's own citation in the answer is imprecise.

    `strategy` selects which retrieval strategy to use (falls back to
    settings.retrieval_strategy). Imported lazily here, rather than at
    module top level, because document_reviewer.tools imports
    RETRIEVE_TOP_K/format_docs from this module — importing tools at
    the top of this file would create a circular import.

    Strategies in CHAIN_STRATEGIES (e.g. agentic) don't fit the
    retrieve-then-generate shape below — they run their own full
    question -> answer loop, so we delegate to their build_chain()
    directly instead of composing retrieve_step | generate_step.
    """
    from document_reviewer.tools import (
        CHAIN_STRATEGIES,
        SEARCH_TOOL_STRATEGIES,
        get_chain_module,
        get_search_module,
        get_vector_store_for_strategy,
    )

    strategy = strategy or settings.retrieval_strategy

    if strategy in CHAIN_STRATEGIES:
        vector_store = get_vector_store_for_strategy(strategy)
        return get_chain_module(strategy).build_chain(vector_store)

    if strategy not in SEARCH_TOOL_STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    vector_store = get_vector_store_for_strategy(strategy)
    search = get_search_module(strategy).search
    llm = get_llm()

    # Step A: retrieve once via the selected strategy, keeping both the
    # raw docs (for citation) and the formatted text (for the prompt)
    # alongside the original question.
    def _retrieve(question: str) -> dict:
        context, source_documents = search(vector_store, question)
        return {"question": question, "context": context, "source_documents": source_documents}

    retrieve_step = RunnableLambda(_retrieve)

    # Step B: generate the answer from {context} + {question}, while
    # passing source_documents through unchanged for the final output.
    generate_step = RunnableParallel(
        answer=RAG_PROMPT | llm | StrOutputParser(),
        source_documents=RunnableLambda(lambda x: x["source_documents"]),
    )

    return retrieve_step | generate_step


if __name__ == "__main__":
    from pathlib import Path

    chain = build_rag_chain()

    questions = [
        "How many vacation days do I get?",
        "Is the Widget Pro waterproof?",
        "Does the Widget Pro support Bluetooth?",  # not in our docs
    ]
    for question in questions:
        result: RagResult = chain.invoke(question)
        sources = sorted({Path(d.metadata["source"]).name for d in result["source_documents"]})
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print(f"   (retrieved from: {', '.join(sources)})\n")
