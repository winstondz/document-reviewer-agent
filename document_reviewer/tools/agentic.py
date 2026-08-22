"""Agentic retrieval: instead of always retrieving exactly once, the LLM
gets the retriever wrapped as a tool and decides for itself whether (and
how many times) to call it before answering.

Unlike vector/hybrid/rerank/contextual, this isn't a single search()
call the generation half of rag_chain.py can wrap — the tool-calling
loop *is* the whole interaction, question in, answer out. So this
module exposes build_chain(vector_store) instead of search()/build_tool()
(see tools/__init__.py docstring for the distinction).

Implemented as an explicit bounded loop (not langgraph / create_react_agent)
to keep this dependency-light and keep the iteration cap trivially
auditable: call the LLM, execute any tool calls it made, repeat until it
stops calling tools or a hard cap is hit.
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from document_reviewer.config.prompts import DOCUMENT_ASSISTANT_RULES
from document_reviewer.llm import get_llm
from document_reviewer.rag_chain import extract_text_content, format_docs

# Hard caps — the loop must physically stop at these, not just be told to.
MAX_AGENT_ITERATIONS = 3  # cap on tool-calling loop turns
MAX_SEARCHES_PER_QUESTION = 3  # cap on retriever calls within one question
AGENT_MAX_TOKENS = 1024  # cap per LLM call in the loop

AGENT_SYSTEM_PROMPT = f"""You are a document review assistant. Answer the \
user's question using ONLY information you retrieve via the \
search_documents tool — do not use outside knowledge.

Call search_documents when you need information. You may call it more \
than once with refined queries if the first search doesn't return what \
you need, up to a limit enforced by the tool itself. Once you have \
enough information (or the tool tells you the search limit is reached), \
answer directly without calling the tool again.

Rules:
{DOCUMENT_ASSISTANT_RULES}"""


def build_chain(vector_store):
    def _run(question: str) -> dict:
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        search_call_count = {"count": 0}  # closure-scoped, reset per invocation
        source_documents: list[Document] = []

        @tool
        def search_documents(query: str) -> str:
            """Search the user's documents for relevant passages. Use this
            when you need information to answer the question. You may call
            this tool multiple times with refined queries if the first
            search doesn't return what you need, up to a limit."""
            if search_call_count["count"] >= MAX_SEARCHES_PER_QUESTION:
                return "Search limit reached. Answer with what you have."
            search_call_count["count"] += 1
            docs = retriever.invoke(query)
            source_documents.extend(docs)
            return format_docs(docs)

        base_llm = get_llm().bind(max_tokens=AGENT_MAX_TOKENS)
        llm_with_tools = base_llm.bind_tools([search_documents])

        messages = [SystemMessage(AGENT_SYSTEM_PROMPT), HumanMessage(question)]
        total_input_tokens = 0
        total_output_tokens = 0
        llm_call_count = 0
        response = None

        for _ in range(MAX_AGENT_ITERATIONS):
            response = llm_with_tools.invoke(messages)
            llm_call_count += 1
            usage = response.usage_metadata or {}
            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)
            messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                messages.append(search_documents.invoke(tool_call))
        else:
            # Hit MAX_AGENT_ITERATIONS while the model still wanted to call
            # tools — force one more call with the tool removed from
            # bind_tools so it has no choice but to answer from what it has.
            response = base_llm.invoke(messages)
            llm_call_count += 1
            usage = response.usage_metadata or {}
            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)

        print(
            f"Agentic strategy: {llm_call_count} LLM call(s), "
            f"{total_input_tokens} input / {total_output_tokens} output tokens."
        )

        return {"answer": extract_text_content(response.content), "source_documents": source_documents}

    return RunnableLambda(_run)
