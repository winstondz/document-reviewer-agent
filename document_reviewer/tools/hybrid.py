"""Hybrid retrieval: BM25 keyword search + vector search, combined.

BM25 is a classic keyword-matching algorithm (scores documents by term
overlap with the query, roughly "how many of the query words appear,
weighted by how rare/important those words are"). It catches exact
terms, codes, and quoted phrases that semantic similarity can miss.
Combining it with the existing vector search gives the best of both.
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool

from document_reviewer.rag_chain import RETRIEVE_TOP_K, format_docs


def _build_ensemble_retriever(vector_store):
    # BM25 is a keyword-frequency algorithm, not a similarity search over
    # embeddings — it needs the raw document text to count term overlap,
    # not the vector store. Pull the original chunks back out of Chroma.
    all_docs = vector_store.get()
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = RETRIEVE_TOP_K

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})

    # EnsembleRetriever runs both retrievers, then merges their ranked
    # lists via Reciprocal Rank Fusion (weighted by `weights`) into one
    # combined ranking, rather than picking one retriever's results over
    # the other's.
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],  # equal weight; make tunable later if desired
    )


def search(vector_store, query: str) -> tuple[str, list[Document]]:
    docs = _build_ensemble_retriever(vector_store).invoke(query)
    return format_docs(docs), docs


def build_tool(vector_store):
    @tool(response_format="content_and_artifact")
    def search_hybrid(query: str) -> tuple[str, list[Document]]:
        """Search using both keyword matching and semantic similarity.
        Best when the question contains exact terms, codes, quoted
        phrases, or specific names that plain semantic search might miss."""
        return search(vector_store, query)

    return search_hybrid
