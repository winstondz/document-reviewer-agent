"""The chat model: Claude, wrapped by LangChain's standard chat-model
interface (see app.py's earlier smoke test for why we go through
langchain-anthropic instead of the raw anthropic SDK — composability
with the rest of the chain, added next).
"""

from langchain_anthropic import ChatAnthropic

from document_reviewer.config import settings


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.chat_model,
        api_key=settings.anthropic_api_key,
    )
