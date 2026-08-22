"""Prompt templates for the RAG chain.

Kept separate from the chain logic (rag_chain.py, next) so the actual
wording — the thing you'll want to iterate on most — isn't buried in
code that wires retrievers and models together.
"""

from langchain_core.prompts import ChatPromptTemplate

# Shared grounding rules: reused by every strategy's answer-generation
# prompt (currently this one, and agentic.py's AGENT_SYSTEM_PROMPT,
# which retrieves via a tool instead of {context} injection but must
# follow the same rules) so they can't silently drift apart.
DOCUMENT_ASSISTANT_RULES = """- If the answer isn't available, say so plainly — do not guess or use \
outside knowledge.
- Cite which source file(s) you drew the answer from.
- Be concise and direct."""

# {context} is filled in with the retrieved chunks (see rag_chain.py),
# {question} with the user's question. Telling the model to say so when
# it doesn't know, rather than guessing, is what keeps answers grounded
# in the documents instead of Claude's general training knowledge.
RAG_SYSTEM_PROMPT = f"""You are a document review assistant. Answer the \
user's question using ONLY the context below, which was retrieved from \
the user's own documents.

Rules:
{DOCUMENT_ASSISTANT_RULES}

Context:
{{context}}"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)
