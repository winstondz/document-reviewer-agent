"""Rerank retrieval: cast a wide net with vector search, then use a
cross-encoder to re-score and keep only the most relevant chunks.

A cross-encoder is a different kind of model than the embedding model
used elsewhere: instead of encoding the query and each chunk separately
and comparing vectors, it takes the (query, chunk) pair *together* as
one input and directly scores how relevant the chunk is to that
specific query. That's much more accurate than vector similarity, but
also much slower — you can't precompute anything, every candidate has
to be scored against the query at request time. So the pattern is:
vector search retrieves broadly (cheap, approximate), the cross-encoder
reranks a small candidate set (expensive, precise).
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.tools import tool

from document_reviewer.rag_chain import format_docs

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local, no API key
RETRIEVE_BROAD_K = 15  # cast a wider net before reranking
RETRIEVE_FINAL_K = 4  # keep this many after reranking


def _build_reranking_retriever(vector_store):
    broad_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_BROAD_K})
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=RETRIEVE_FINAL_K)
    # ContextualCompressionRetriever wraps a base retriever with a
    # post-processing step ("compressor") — here it's not shrinking text,
    # just filtering/reordering the candidate list down to top_n.
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=broad_retriever,
    )


def search(vector_store, query: str) -> tuple[str, list[Document]]:
    docs = _build_reranking_retriever(vector_store).invoke(query)
    return format_docs(docs), docs


def build_tool(vector_store):
    @tool(response_format="content_and_artifact")
    def search_rerank(query: str) -> tuple[str, list[Document]]:
        """Search broadly, then re-rank results for relevance before
        returning the best matches. Higher precision than plain search,
        useful when the top result from plain search often feels
        slightly off-target."""
        return search(vector_store, query)

    return search_rerank
