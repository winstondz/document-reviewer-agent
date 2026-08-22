"""The baseline retrieval strategy: plain vector similarity search.

Moved here unchanged from rag_chain.py — see the tools package docstring
for why strategies live as search()/build_tool() pairs.
"""

from langchain_core.documents import Document
from langchain_core.tools import tool

from document_reviewer.rag_chain import RETRIEVE_TOP_K, format_docs


def search(vector_store, query: str) -> tuple[str, list[Document]]:
    # Embeds the query and finds the k nearest chunks by vector distance
    # (semantic similarity) — no keyword matching involved.
    docs = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K}).invoke(query)
    return format_docs(docs), docs


def build_tool(vector_store):
    # search_vector closes over vector_store, so once built it's a
    # self-contained callable an LLM can invoke via bind_tools() without
    # needing the vector store passed in as an argument each time.
    @tool(response_format="content_and_artifact")
    def search_vector(query: str) -> tuple[str, list[Document]]:
        """Search the documents using standard semantic similarity.
        Good general-purpose default when no other strategy clearly fits."""
        # content_and_artifact: the first return value (str) is what the
        # LLM sees as the tool result; the second (the Document list) is
        # the "artifact" — kept for our own citation logic, invisible to
        # the LLM.
        return search(vector_store, query)

    return search_vector
