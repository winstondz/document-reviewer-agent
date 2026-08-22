"""Central, typed app configuration.

Values are read from environment variables, or from a .env file in the
project root if present (see .env.example for the expected keys).
"""

from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings' env_file=".env" below only populates THIS Settings
# object's own fields — it does not inject values into os.environ. That's
# fine for settings this module passes through explicitly (e.g.
# anthropic_api_key -> ChatAnthropic(api_key=...) in llm.py), but
# LangSmith's tracing hooks read LANGSMITH_* directly from os.environ,
# with no explicit code path carrying them there. Loading .env into the
# real process environment here is what makes tracing actually turn on.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    chat_model: str = "claude-sonnet-5"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "documents"
    contextual_collection_name: str = "documents_contextual"
    graph_store_path: str = "graph_store.pkl"
    retrieval_strategy: str = "vector"
    # "llm" is reserved for a future LLMRouter (see router.py) — not
    # implemented yet, but validated against here so this config
    # surface doesn't need to change again once it is.
    router_type: Literal["heuristic", "llm"] = "heuristic"

    # LangSmith tracing — optional, off by default. Every .invoke() on
    # an LCEL runnable (every chain in this project) is traced
    # automatically once these are set; no code changes needed in
    # rag_chain.py, tools/, or router.py. These fields exist purely so
    # they're validated/typed like everything else — the LangSmith SDK
    # itself reads them straight from os.environ (see load_dotenv above),
    # not from this Settings object.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "document-reviewer-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"


settings = Settings()
