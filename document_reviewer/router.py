"""Automated mode: instead of the user always passing --strategy, a
router picks the strategy for a given question.

Router is a Protocol (structural typing — anything with a matching
select_strategy method satisfies it, no inheritance required) so a
smarter router can be dropped in later without touching call sites.
HeuristicRouter is the only implementation today; LLMRouter is a stub
for later, once every search strategy is already a callable tool (see
tools/__init__.py) — binding those tools and letting the model pick by
calling one becomes small to build at that point.

`available_strategies` must be computed by the caller (app.py / the
future API layer), from whichever collections/graphs actually exist on
disk — the router itself is never responsible for checking that, so
the constraint is enforced in exactly one place. See
tools.get_available_strategies().
"""

import re
from typing import Protocol


class Router(Protocol):
    def select_strategy(self, question: str, available_strategies: list[str]) -> str:
        """Return the name of the strategy to use for this question.
        Must only return a name present in available_strategies."""
        ...


class HeuristicRouter:
    """Rule-based dispatch, no LLM call — cheap and deterministic.
    These specific rules are a starting point, not gospel; adjust once
    real question patterns are visible."""

    def select_strategy(self, question: str, available_strategies: list[str]) -> str:
        def available(name: str) -> bool:
            return name in available_strategies

        # Quoted phrase or code-like token -> keyword search matters
        if available("hybrid") and (re.search(r'"[^"]+"', question) or re.search(r"\b[A-Z]{2,}\d+\b", question)):
            return "hybrid"

        # Multiple entities / "and" / "compare" / "relationship" -> graph
        if available("graphrag") and re.search(r"\b(compare|relationship|between|and how)\b", question, re.I):
            return "graphrag"

        # Long or multi-part question -> let the agent decide how much to search
        if available("agentic") and (len(question.split()) > 20 or "?" in question[:-1]):
            return "agentic"

        # Default: best general-purpose option that exists
        for fallback in ("rerank", "hybrid", "vector"):
            if available(fallback):
                return fallback
        return "vector"


class LLMRouter:
    """Not implemented in this pass. Will bind the four search-tool
    strategies' build_tool(vector_store) objects — whichever have an
    index available — via llm.bind_tools([...]) and let the model pick
    by calling one. Must still respect available_strategies. Add later,
    once HeuristicRouter's rules have been validated against real
    usage — see PLAN.md for why this is deferred despite being cheap
    to build."""

    def select_strategy(self, question: str, available_strategies: list[str]) -> str:
        raise NotImplementedError("Phase 2.5 — not part of this plan")


def get_router(router_type: str) -> Router:
    if router_type == "heuristic":
        return HeuristicRouter()
    if router_type == "llm":
        return LLMRouter()
    raise ValueError(f"Unknown router_type: {router_type}")
