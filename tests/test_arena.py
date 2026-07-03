"""Tests for arena.py — the evaluation harness over engine.run_game."""

from arena import POLICIES, run_match


def up(_state):
    return "up"


def test_registry_contains_algo_and_hybrid():
    assert {"algo", "hybrid"} <= set(POLICIES)
    assert all(callable(p) for p in POLICIES.values())


def test_run_match_structure_and_alternation():
    # Both policies just run up and die on the top wall; whoever starts lower
    # survives longer. Alternating starts means each side wins one game.
    result = run_match(up, up, games=2, width=11, height=11, seed=0)

    assert result["games"] == 2
    assert result["wins_a"] + result["wins_b"] + result["draws"] == 2
    assert result["wins_a"] == 1 and result["wins_b"] == 1

    for side in ("a", "b"):
        assert result["deaths"][side].get("wall", 0) == 1
        assert result["turns"][side]["avg"] > 0
        assert result["latency_ms"][side]["p50"] >= 0.0
        assert result["latency_ms"][side]["p95"] >= result["latency_ms"][side]["p50"]
