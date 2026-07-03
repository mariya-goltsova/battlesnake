"""Pipeline tests for policy.py: legality, safety, fallback, parity, latency."""

import random
import time

import pytest

from src_rl import policy, sim
from src_rl.features import FEATURE_NAMES, build_context, candidate_features
from src_rl.safety import legal_moves
from src_rl.tests.helpers import make_snake, make_state


def test_model_features_are_a_subset_of_feature_names():
    assert set(policy._MODEL["feature_names"]) <= set(FEATURE_NAMES)


def test_choose_move_is_legal_and_deterministic():
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)])
    enemy = make_snake("e", [(2, 2), (2, 3), (2, 4), (2, 5)])
    state = make_state(me, snakes=[me, enemy], food=[(9, 9)])
    move = policy.choose_move(state)
    assert move in legal_moves(state)
    assert all(policy.choose_move(state) == move for _ in range(3))


def test_never_picks_avoidable_certain_death():
    # left = 1-cell trap, right = open space (from the safety test suite).
    me = make_snake("me", [(1, 0), (1, 1), (1, 2), (2, 2)])
    wall = make_snake("w", [(0, 1), (0, 2), (0, 3)])
    state = make_state(me, snakes=[me, wall])
    assert policy.choose_move(state) == "right"


def test_never_picks_avoidable_losing_h2h():
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)])
    enemy = make_snake("e", [(7, 5), (8, 5), (9, 5), (9, 6)])  # longer, head (7,5)
    state = make_state(me, snakes=[me, enemy])
    assert policy.choose_move(state) != "right"


def test_cornered_returns_least_bad_not_crash():
    me = make_snake("me", [(0, 0), (1, 0), (1, 1), (0, 1), (0, 1)])
    state = make_state(me, width=2, height=2)
    assert policy.choose_move(state) in {"up", "down", "left", "right"}


def test_fallback_on_scorer_exception(monkeypatch):
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)])
    state = make_state(me, food=[(0, 0)])

    def boom(*a, **k):
        raise RuntimeError("model broke")

    monkeypatch.setattr(policy, "rank_moves", boom)
    assert policy.choose_move(state) in legal_moves(state)


def test_scorer_parity_with_manual_dot_product():
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)], health=40)
    enemy = make_snake("e", [(2, 2), (2, 3), (2, 4)])
    state = make_state(me, snakes=[me, enemy], food=[(1, 1), (9, 9)])
    ctx = build_context(state)
    m = policy._MODEL
    for move in legal_moves(state):
        feats = candidate_features(state, move, ctx)
        manual = m["intercept"] + sum(
            m["coef"][i] * ((feats.get(n, 0.0) - m["mean"][i]) / m["std"][i] if m["std"][i] else 0.0)
            for i, n in enumerate(m["feature_names"])
        )
        assert policy.score_features(feats) == pytest.approx(manual, abs=1e-9)


# --- latency budget -------------------------------------------------------------------


def _rollout_states(n_games=6, max_turns=120):
    """Realistic mid-game states: roll games where every snake uses the policy."""
    states = []
    for seed in range(n_games):
        st = sim.init_game(seed=seed)
        for _ in range(max_turns):
            if sim.game_over(st):
                break
            moves = {}
            for s in st["snakes"]:
                if s["alive"]:
                    gs = sim.make_game_state(st, s["id"])
                    states.append(gs)
                    moves[s["id"]] = policy.choose_move(gs)
            st = sim.step(st, moves)
    return states


def test_choose_move_latency_p95_under_50ms():
    states = _rollout_states()
    assert len(states) >= 200
    sample = random.Random(0).sample(states, 200)
    timings = []
    for gs in sample:
        t0 = time.perf_counter()
        policy.choose_move(gs)
        timings.append(time.perf_counter() - t0)
    timings.sort()
    p95 = timings[int(len(timings) * 0.95)]
    assert p95 < 0.050, f"p95 latency {p95 * 1000:.1f}ms exceeds 50ms budget"
