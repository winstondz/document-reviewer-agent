# Retrieval Strategies

This project can answer questions using six different ways of finding
relevant document content, selectable with `--strategy` on the CLI
(`vector`, `hybrid`, `rerank`, `contextual`, `agentic`, `graphrag`).
This doc explains what each one actually does, in plain language, with
examples from the sample docs (`vacation_policy.txt`, `product_faq.md`
in `data/`). The five implemented strategies are covered first; `graphrag`
is the sixth and, as of this writing, is not yet built.

---

## 1. `vector` (the baseline)

Every chunk of your documents gets converted into a list of numbers — a
"vector" — positioned in space based on what the text *means*, not what
words it literally contains. Chunks about similar topics end up near
each other even with zero words in common. Your question gets converted
the same way, and the system just returns the 4 chunks positioned
closest to it.

**Example**: "How much time off do I get?" retrieves the vacation
policy chunk even though it never uses the words "time off" — the
model recognizes "time off" and "vacation accrual" as the same idea.

**Weakness**: exact strings, codes, and names don't have much
"meaning" for the model to place precisely — a query for `IP67` can
get lost among chunks that are thematically similar but don't actually
contain that term.

**Best for**: general-purpose default, works well when you don't have
a stronger reason to pick something else.

**The math**: "closeness" is cosine similarity — the angle between the
question's vector and a chunk's vector, ignoring their length:
```
cos(θ) = (A · B) / (|A| × |B|)
```
`A · B` is the dot product (multiply each matching pair of numbers,
sum the results); `|A|` and `|B|` are each vector's length. Result
ranges -1 to 1, where 1 means "pointing in exactly the same
direction" — very similar meaning. Chroma computes this between the
question and every stored chunk, and returns the highest scores.

---

## 2. `hybrid` (vector + keyword search, combined)

Runs two searches at once and merges the results:
- The vector search above (meaning-based).
- Old-school **keyword search**: count which words overlap between
  your question and each chunk, weighted so rare/distinctive words
  count for a lot and common words count for nothing.

**Example**: "Is the Widget Pro rated IP67?" — the keyword half
catches the exact term `IP67` even if the vector half's ranking is
lukewarm on it.

**Best for**: questions with exact terms, product codes, quoted
phrases, or specific names — anything where the exact words matter,
not just the gist.

**The math**: the keyword half uses **BM25**, which scores each chunk
on term frequency (how often a query word appears, with diminishing
returns per repeat), inverse document frequency (rare words score
high, common words score near zero), and length normalization (a
short chunk mentioning the word counts more than a long one mentioning
it once amid unrelated text):
```
score = Σ IDF(word) × (TF(word) × (k+1)) / (TF(word) + k × (1 - b + b × chunk_length / avg_length))
```
The two ranked lists (keyword + vector) are then merged with
**Reciprocal Rank Fusion**:
```
score = Σ 1 / (k + rank)
```
summed across every retriever that returned the chunk, where `rank` is
its position in that retriever's list and `k` is a small constant. A
chunk ranked #1 by keyword search and #4 by vector search can outscore
one ranked #2 by vector search alone.

---

## 3. `rerank` (search broad, then double-check with a stricter judge)

Two steps:
1. Vector search, but ask for a lot more candidates than usual (15
   instead of 4) — cast a wide net.
2. A second, slower, much more careful model looks at the question and
   *each* candidate chunk together, and scores how relevant that
   specific pairing really is. Keep only the top 4 it rates highest.

Think of it like a librarian doing a fast first pass to pull anything
plausibly related, followed by an expert who actually reads each
candidate and throws out the weak ones.

**Example**: "What happens if my Widget Pro gets water damage?" — the
initial 15 candidates might include the waterproofing section, the
warranty section, and the reset instructions; the second pass ranks
the warranty section (which specifically says water damage isn't
covered) above the others.

**Best for**: when the default search's top result often feels "close
but not quite right" — trades a little speed for better precision.

**The math**: step 1 is the same bi-encoder cosine similarity as
`vector` — question and chunk are each converted to a vector
*independently*, then compared. Step 2 uses a **cross-encoder**: the
question and chunk are fed into the model *together*, as one combined
input, and it directly outputs a single relevance score for that exact
pairing — no cosine similarity involved, just a learned score. This is
more accurate, because the model can notice interactions between the
two, but nothing can be precomputed ahead of time, which is why it
only runs on a small shortlist (15 candidates) instead of the whole
document set.

---

## 4. `contextual` (annotate chunks before storing them)

This changes what happens *before* you ever ask a question — at
indexing time, not search time.

**The problem**: chopping a document into small chunks can strip away
context. A chunk that just says "It is rated IP67" is fine when read
in the full document, but on its own, "It" doesn't say what's being
discussed — that ambiguity can hurt the vector search's ability to
recognize the chunk is relevant.

**The fix**: before storing each chunk, an LLM reads the whole
document plus that one chunk, and writes a short blurb like "This
chunk is from the Widget Pro FAQ's waterproofing section," which gets
glued onto the front of the chunk before it's turned into a vector.

**Example**: the raw chunk "It is rated IP67, meaning it is dust-tight
and can withstand immersion..." becomes "This chunk is from the Widget
Pro FAQ, describing its water and dust resistance rating. It is rated
IP67, meaning..." — now findable on its own merits.

**Cost to know about**: one LLM call *per chunk*, at ingest time only.
Fine for a handful of chunks; for a large document set this adds up,
which is why the code enforces a hard chunk-count limit before
proceeding.

**Best for**: document sets where individual chunks lose important
context when separated from their source (pronouns, "the above,"
section-specific jargon).

**The math**: none new — retrieval here is the identical cosine
similarity from `vector`. The only change is the *input* to the
embedding step: `embed(blurb + chunk_text)` instead of
`embed(chunk_text)`. Since the embedding is a function of the whole
input text, prepending context words shifts where the resulting vector
lands in that space, ideally closer to where a well-formed question
about that content would land.

---

## 5. `agentic` (let the AI decide how much to search)

Every strategy above searches exactly once per question. This one
hands the AI a "search" button it can press for itself, as many times
as it wants (within a limit), and lets it decide.

For a simple question it might search once and answer immediately. For
a vague or multi-part question, it might search, notice the results
don't fully cover it, refine the query, and search again — the way a
person doing research re-searches after a first attempt falls short.

**Example**: "What's the vacation policy and is the Widget Pro
waterproof?" — the model may search for the vacation policy, then
separately search for waterproofing, combining both into one answer.

**Cost to know about**: each loop iteration is a full extra AI call,
so cost scales with how many times it decides to search. The code
enforces hard caps on iterations and searches per question so it can't
run away.

**Best for**: broad, vague, or multi-part questions where you don't
know in advance how many searches it'll take to answer well.

**The math**: no new retrieval math — each individual search call is
the same cosine-similarity vector search as `vector`. The only new
mechanism is control flow: cost grows as
`O(searches × LLM calls per search round-trip)` instead of the fixed
`O(1)` search of every other strategy, bounded above by
`MAX_AGENT_ITERATIONS × MAX_SEARCHES_PER_QUESTION`.

---

## 6. `graphrag` (build a knowledge graph, then traverse it)

Instead of storing individual chunks, this strategy extracts
**entities and relationships** from your documents into a graph at
ingest time — e.g. `Widget Pro` —rated— `IP67`, `Full-time employee`
—accrues— `15 days vacation`. To answer a question, it doesn't do a
similarity search at all — it finds the relevant entities mentioned in
the question, walks the graph to pull in connected facts, and hands
those to the LLM.

**Example**: "How does the Widget Pro's warranty relate to its water
resistance rating?" — a pure chunk-similarity search might return the
warranty section *or* the waterproofing section, but graph traversal
can follow the actual `covers`/`excludes` edges connecting `IP67
rating` to `warranty exclusions`, surfacing the specific relationship
rather than two separately-ranked chunks.

**Cost to know about**: the most expensive of the six — extraction is
one LLM call per chunk at ingest (like `contextual`, but producing
structured entities/relations instead of a blurb), plus additional LLM
calls at query time to identify which entities the question is asking
about. A hard chunk-count ceiling is checked *before* any extraction
calls run, so a large document set fails fast with a clear error
rather than silently burning through calls.

**Best for**: questions that span multiple related facts —
comparisons, "how does X relate to Y," multi-hop reasoning — rather
than "find the one passage that answers this."

**The math**: no embeddings or similarity scoring involved at query
time — it's graph traversal (walking nodes connected by edges, e.g.
breadth-first search out from the entities mentioned in the question,
out to some depth). The "learning" happens once, at ingest, when the
LLM reads each chunk and outputs structured JSON
(`{"entities": [...], "relations": [...]}`) that becomes graph nodes
and edges.

---

## Comparison

| Strategy | Core mechanism | Best for | Relative cost |
|---|---|---|---|
| `vector` | Meaning-based similarity search | General-purpose default | Lowest |
| `hybrid` | Vector search + keyword search, merged | Exact terms, codes, quoted phrases | Low |
| `rerank` | Broad vector search, then precision re-scoring | When top results feel slightly off-target | Low–medium |
| `contextual` | Chunks pre-annotated with context before embedding | Chunks that lose meaning out of context | Medium (extra ingest-time LLM calls) |
| `agentic` | AI decides how many times to search | Vague or multi-part questions | High (multiple LLM calls per question) |
| `graphrag` | Entity/relationship graph, traversed at query time | Multi-fact comparisons, relationships between entities | Highest (ingest + query-time LLM calls) |

---

## Where this fits in the broader landscape

### More techniques used in production

- **Query rewriting / multi-query retrieval** — the LLM rewrites your question into several variants before searching (e.g. "vacation days" → "PTO policy," "time off accrual," "leave entitlement"), runs all of them, merges results. Helps when a user's phrasing doesn't match the document's phrasing.
- **HyDE (Hypothetical Document Embeddings)** — instead of embedding the question, the LLM first writes a *hypothetical answer*, then embeds that and searches with it. Answers tend to be closer in "meaning-space" to real answer chunks than questions are.
- **Parent-document / "small-to-big" retrieval** — search over small, precise chunks, but return the larger parent section they came from, so the LLM gets more surrounding context without needing a bigger initial chunk size.
- **RAPTOR** — builds a tree of summaries over the document (cluster chunks, summarize each cluster, repeat), so a query can match either a fine-grained chunk or a high-level summary depending on how broad the question is.
- **Late-interaction / ColBERT-style retrieval** — a middle ground between bi-encoder and cross-encoder: keeps per-token embeddings instead of one vector per chunk, giving much of the cross-encoder's precision without needing a separate reranking pass.
- **Self-RAG / corrective RAG** — the model critiques its own retrieved chunks ("are these actually relevant?") and re-retrieves or falls back to a different strategy if not — a more general, less rule-based version of what this project's `HeuristicRouter` (Phase 2) does.

This project already covers the two most commonly deployed real-world patterns — hybrid search and rerank — which is what most production RAG stacks converge on before reaching for anything fancier.

### Model providers — what they ship natively

- **Anthropic** — originated and published the `contextual` technique used in this project ("Contextual Retrieval," 2024). Ships prompt caching (cheaper reuse of large repeated context) and the Model Context Protocol (MCP), which standardizes how Claude connects to external data sources/tools — closer to this project's `agentic` pattern than to a hosted search index. No hosted vector search product of its own.
- **OpenAI** — ships a hosted `file_search` tool in the Assistants/Responses API: upload files, OpenAI manages chunking + embedding + vector search + reranking for you, no separate vector DB needed. The closest thing to a fully managed version of what's built by hand in this project.
- **Google (Gemini)** — Vertex AI Search (enterprise-grade managed retrieval), Grounding with Google Search (retrieval from the live web instead of your documents), and a Vertex AI RAG Engine for custom pipelines. Gemini also leans on its very large context window as a retrieval-avoidance strategy — for smaller corpora, "just stuff the whole document set into context" can beat retrieval entirely.
- **AWS** — several native options: **Amazon Bedrock Knowledge Bases**, a fully managed RAG pipeline (chunking, embedding, vector store, retrieval, and citations, wired directly into Bedrock model calls) — the most direct AWS equivalent to OpenAI's `file_search`. **Amazon Kendra**, an enterprise search service with its own ML-based relevance ranking, often used as a retrieval backend for Bedrock. **Amazon OpenSearch Service**, which has a k-NN plugin for vector similarity search (used as the underlying vector store by some Bedrock Knowledge Base configurations). And **Amazon Q Business**, a higher-level packaged product for enterprise Q&A over internal data, built on top of these.
- **Cohere** — provides a hosted Rerank API, the same cross-encoder idea as this project's `rerank` strategy but as a paid API call instead of a local model — very commonly bolted onto other companies' RAG stacks specifically for that one step.
- **Vector databases** (Pinecone, Weaviate, Qdrant, Milvus) increasingly ship hybrid search and reranking as built-in features rather than something you assemble yourself, the way this project does with Chroma.
