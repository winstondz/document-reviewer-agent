"""Unit tests for HeuristicRouter — pure logic, no LLM/API calls needed."""

import pytest

from document_reviewer.router import HeuristicRouter, LLMRouter, get_router

ALL_STRATEGIES = ["vector", "hybrid", "rerank", "contextual", "agentic", "graphrag"]


@pytest.fixture
def router():
    return HeuristicRouter()


def test_quoted_phrase_routes_to_hybrid(router):
    assert router.select_strategy('What does "IP67" mean?', ALL_STRATEGIES) == "hybrid"


def test_code_like_token_routes_to_hybrid(router):
    assert router.select_strategy("Is the Widget Pro rated IP67?", ALL_STRATEGIES) == "hybrid"


def test_relationship_question_routes_to_graphrag(router):
    assert router.select_strategy("What is the relationship between vacation and tenure?", ALL_STRATEGIES) == "graphrag"


def test_compare_question_routes_to_graphrag(router):
    assert router.select_strategy("Compare the warranty and the return policy.", ALL_STRATEGIES) == "graphrag"


def test_long_question_routes_to_agentic(router):
    question = " ".join(["word"] * 21)
    assert router.select_strategy(question, ALL_STRATEGIES) == "agentic"


def test_multi_question_sentence_routes_to_agentic(router):
    question = "What is the vacation policy? Is the Widget Pro waterproof?"
    assert router.select_strategy(question, ALL_STRATEGIES) == "agentic"


def test_default_falls_back_to_rerank_when_available(router):
    assert router.select_strategy("How many vacation days do I get?", ALL_STRATEGIES) == "rerank"


def test_fallback_chain_skips_unavailable_strategies(router):
    available = ["vector", "hybrid"]
    assert router.select_strategy("How many vacation days do I get?", available) == "hybrid"

    available = ["vector"]
    assert router.select_strategy("How many vacation days do I get?", available) == "vector"


def test_never_selects_a_strategy_outside_available_even_when_rule_matches(router):
    # Quoted phrase would normally route to hybrid, but hybrid isn't
    # available here — must fall through to the fallback chain instead.
    available = ["vector", "rerank"]
    assert router.select_strategy('What does "IP67" mean?', available) == "rerank"


def test_get_router_returns_heuristic_router():
    assert isinstance(get_router("heuristic"), HeuristicRouter)


def test_get_router_returns_llm_router_stub():
    assert isinstance(get_router("llm"), LLMRouter)


def test_llm_router_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        LLMRouter().select_strategy("anything", ALL_STRATEGIES)


def test_get_router_rejects_unknown_type():
    with pytest.raises(ValueError):
        get_router("something-else")
