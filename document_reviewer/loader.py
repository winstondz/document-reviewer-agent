"""Turn files in a directory into LangChain Document objects.

A Document is just page_content (the raw text) + metadata (e.g. which file
it came from). Everything downstream — splitting, embedding, retrieval —
operates on this shape, regardless of what the original file format was.

For now we handle plain-text formats (.txt, .md) with TextLoader. PDF and
.docx support can be added later as extra loader_cls entries per extension.
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

# extension -> (glob pattern, loader class, loader kwargs)
LOADERS_BY_EXTENSION = {
    ".txt": "*.txt",
    ".md": "*.md",
}


def load_documents(directory: str) -> list[Document]:
    """Load every supported file under `directory` into Document objects."""
    docs: list[Document] = []
    for glob_pattern in LOADERS_BY_EXTENSION.values():
        loader = DirectoryLoader(
            directory,
            glob=glob_pattern,
            loader_cls=TextLoader,
        )
        docs.extend(loader.load())
    return docs


if __name__ == "__main__":
    loaded = load_documents("data")
    print(f"Loaded {len(loaded)} document(s):")
    for doc in loaded:
        source = Path(doc.metadata["source"]).name
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  - {source} ({len(doc.page_content)} chars): {preview}...")
