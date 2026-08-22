"""Contextual retrieval (Anthropic's "Contextual Retrieval" technique):
before embedding each chunk, prepend a short LLM-generated blurb that
situates it within its source document.

Why this helps: a chunk taken out of context can lose the information
that made it findable. E.g. a chunk that just says "It is rated IP67"
loses the fact that "it" is the Widget Pro once separated from its
document — a query for "Widget Pro water resistance" might not match
the chunk's own text well. Prepending "This chunk is from the Widget
Pro FAQ's waterproofing section." gives the embedding model that
context back, at the cost of one extra LLM call per chunk at ingest
time (query time is unaffected — this only changes what gets embedded).

This strategy changes ingestion, not just retrieval, so unlike
vector/hybrid/rerank it needs its own vector store collection (the
embedded text is different, so it must not share a collection with
strategies embedding the raw chunk text).
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from document_reviewer.config import settings
from document_reviewer.llm import get_llm
from document_reviewer.rag_chain import format_docs
from document_reviewer.vector_store import build_vector_store
from document_reviewer.vector_store import get_vector_store as _get_base_vector_store

# Hard cap on chunks contextualize_chunks() will process without --force.
# This makes ONE LLM call per chunk — fine for our 8 sample chunks, but
# does NOT scale silently: a real 1000-chunk document set would mean
# 1000 LLM calls. Raise this deliberately (or pass force=True) once
# you've accounted for that cost.
MAX_CONTEXTUAL_CHUNKS = 200

CONTEXTUAL_SYSTEM_PROMPT = """You are helping index a document for search. \
Given the full document and one chunk from it, write a short (1-2 \
sentence) context statement that situates the chunk within the \
document, to improve search retrieval of the chunk. Answer only with \
the succinct context, nothing else."""

CONTEXTUAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUAL_SYSTEM_PROMPT),
        ("human", "<document>\n{full_document}\n</document>\n\n<chunk>\n{chunk}\n</chunk>"),
    ]
)


def contextualize_chunks(docs: list[Document], chunks: list[Document], force: bool = False) -> list[Document]:
    """Returns new Document objects with an LLM-generated context blurb
    prepended to each chunk's page_content, before embedding. Makes one
    LLM call per chunk — see the module docstring's cost note.
    """
    if len(chunks) > MAX_CONTEXTUAL_CHUNKS and not force:
        raise ValueError(
            f"Contextualizing {len(chunks)} chunks would require {len(chunks)} LLM calls, "
            f"exceeding the safety limit of {MAX_CONTEXTUAL_CHUNKS}. Pass force=True "
            "(CLI: --force) to proceed anyway."
        )

    llm = get_llm()
    chain = CONTEXTUAL_PROMPT | llm

    # Lookup from source filename -> full original document text, so
    # each chunk can be given its parent document for context.
    full_text_by_source = {doc.metadata["source"]: doc.page_content for doc in docs}

    contextualized = []
    for chunk in chunks:
        full_document = full_text_by_source[chunk.metadata["source"]]
        context_blurb = chain.invoke({"full_document": full_document, "chunk": chunk.page_content}).content
        new_content = f"{context_blurb}\n\n{chunk.page_content}"
        contextualized.append(Document(page_content=new_content, metadata=chunk.metadata))
    return contextualized


def build_vector_store_contextual(docs: list[Document], chunks: list[Document], force: bool = False):
    """Ingestion entry point for the contextual strategy: contextualize
    every chunk, then embed into the separate contextual collection
    (never the default one — the embedded text differs from the raw
    chunk text every other strategy shares).
    """
    enriched_chunks = contextualize_chunks(docs, chunks, force=force)
    return build_vector_store(enriched_chunks, collection_name=settings.contextual_collection_name)


def get_vector_store():
    """Open the contextual collection (not the default one). Named to
    match document_reviewer.vector_store.get_vector_store's shape so
    the registry can call it generically per strategy — see
    tools/__init__.py's get_vector_store_for_strategy."""
    return _get_base_vector_store(collection_name=settings.contextual_collection_name)


def search(vector_store_contextual, query: str) -> tuple[str, list[Document]]:
    docs = vector_store_contextual.as_retriever(search_kwargs={"k": 4}).invoke(query)
    return format_docs(docs), docs


def build_tool(vector_store_contextual):
    @tool(response_format="content_and_artifact")
    def search_contextual(query: str) -> tuple[str, list[Document]]:
        """Search an index where each chunk was pre-annotated with
        context about its place in the source document. Best for
        questions where a chunk's meaning depends on surrounding
        context the plain chunk text alone wouldn't convey."""
        return search(vector_store_contextual, query)

    return search_contextual
