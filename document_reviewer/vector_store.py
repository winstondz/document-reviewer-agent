"""Chroma vector store: persists chunk embeddings to disk and answers
"which chunks are closest in meaning to this query?" via similarity
search (the cosine-similarity math from embeddings.py, done for us by
Chroma's index instead of by hand).

Two entry points, used at different times:
  - build_vector_store: ingest time — embed everything, write to disk.
  - get_vector_store:   query time — open what's already on disk.
Splitting them this way means asking a question never re-embeds the
whole document set; only running ingest again does that.
"""

from __future__ import annotations

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from document_reviewer.config import settings
from document_reviewer.embeddings import get_embeddings


def build_vector_store(chunks: list[Document], collection_name: str | None = None) -> Chroma:
    """Embed `chunks` and persist them to disk, replacing any existing
    collection of the same name. Use this at ingest time.

    `collection_name` defaults to settings.chroma_collection_name.
    Strategies whose chunks aren't compatible with the default
    collection (e.g. contextual, which prepends an LLM-generated blurb
    to each chunk before embedding) pass their own name here so they
    get a separate collection instead of colliding with the default.

    Chroma.from_documents does three things in one call: runs every
    chunk's page_content through the embedding model, stores each
    resulting vector alongside the chunk's original text + metadata
    (so a match can be traced back to its source file), and writes it
    all to `persist_directory` so it survives across process restarts.

    from_documents on its own does NOT clear an existing collection of
    the same name — it appends to it. Calling this repeatedly (every
    ingest run, every test run) would otherwise pile up duplicate
    copies of every chunk, which crowds out genuinely different chunks
    from top-k retrieval results. So we explicitly delete any existing
    collection first, making this call idempotent.
    """
    collection_name = collection_name or settings.chroma_collection_name

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    existing_collections = {c.name for c in client.list_collections()}
    if collection_name in existing_collections:
        client.delete_collection(collection_name)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=settings.chroma_persist_dir,
    )


def get_vector_store(collection_name: str | None = None) -> Chroma:
    """Open an existing persisted collection without re-embedding
    anything. Use this at query time, once ingest has already run.

    `collection_name` defaults to settings.chroma_collection_name — see
    build_vector_store for why a strategy might override it.

    `embedding_function` is still required here (not for the existing
    stored vectors, which are already on disk) — it's needed to embed
    the *new* incoming query text so it can be compared against them.
    """
    return Chroma(
        embedding_function=get_embeddings(),
        collection_name=collection_name or settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
    )
