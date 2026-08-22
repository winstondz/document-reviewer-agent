"""Split loaded Documents into smaller chunks for embedding/retrieval.

RecursiveCharacterTextSplitter tries to break on paragraph boundaries
first, then sentences, then words — only falling back to a hard character
cut if nothing else fits. This keeps chunks as semantically coherent as
possible instead of slicing mid-sentence.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def split_documents(docs: list[Document]) -> list[Document]:
    """Split a list of Documents into smaller chunk Documents.

    Each resulting chunk keeps the original document's metadata (e.g.
    `source`), so we can still trace a chunk back to the file it came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    from pathlib import Path

    from document_reviewer.loader import load_documents

    docs = load_documents("data")
    chunks = split_documents(docs)

    print(f"{len(docs)} document(s) -> {len(chunks)} chunk(s)\n")
    for i, chunk in enumerate(chunks):
        source = Path(chunk.metadata["source"]).name
        preview = chunk.page_content[:70].replace("\n", " ")
        print(f"[{i}] {source} ({len(chunk.page_content)} chars): {preview}...")
