"""Embedding model: turns text into vectors that capture meaning.

Runs locally (no API key, no per-call cost) via sentence-transformers.
This is a separate model from the chat model (Claude) — embeddings and
text generation are different jobs; there's no requirement they share a
provider. The model name is configurable via EMBEDDING_MODEL in .env /
the environment (see config.py) — swap it without touching this file.

--- The math, in plain terms ---

Each piece of text becomes a vector: a list of numbers (384 of them, for
the model we use by default). Think of it as one row in a big table —
one row per chunk, 384 columns. Comparing a query vector against every
stored row at once is exactly a matrix-vector multiplication.

To score how similar two vectors are, we use *cosine similarity*:

  1. Dot product: multiply matching numbers together, then sum them.
     a = [1, 2], b = [2, 3]  ->  dot(a, b) = (1*2) + (2*3) = 8

  2. That raw number is biased by vector length (a "longer" vector can
     score higher just by being bigger, not by being more similar), so
     we normalize by dividing by both vectors' lengths:

       cosine_similarity(a, b) = dot(a, b) / (length(a) * length(b))

     "Length" here is the usual distance formula, just extended to
     however many dimensions the vector has:
       length(v) = sqrt(v[0]^2 + v[1]^2 + ... + v[n]^2)

  3. The result ranges from -1 (opposite meaning) to 1 (same meaning),
     with 0 meaning unrelated. Chroma (our vector store, added next
     step) does this same computation internally, via an index, for
     every stored chunk against the query — that's how retrieval finds
     "the closest meaning" instead of "the closest exact words."
"""

from langchain_huggingface import HuggingFaceEmbeddings

from document_reviewer.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    length_a = sum(x * x for x in a) ** 0.5
    length_b = sum(y * y for y in b) ** 0.5
    return dot_product / (length_a * length_b)
