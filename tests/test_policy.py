"""Tests for policy.py — the hybrid pipeline, and logic.py as a thin wrapper."""

from helpers import make_snake, make_state

import policy
from core import DIRECTIONS
from features import candidate_features
from policy import MODEL, choose_move


def test_returns_legal_move_on_open_board():
    me = make_snake("me", [(5, 5), (5, 4), (5, 3)])
    state = make_state(me, food=[(0, 0)])

    move = choose_move(state)

    assert move in DIRECTIONS
    assert move != "down"  # (5,4) is our neck


def test_h2h_veto_respected_end_to_end():
    me = make_snake("me", [(5, 5), (5, 4), (5, 3)])
    enemy = make_snake("e", [(7, 5), (8, 5), (9, 5), (9, 4)])  # length 4 > 3
    state = make_state(me, snakes=[me, enemy])

    assert choose_move(state) in {"up", "left"}  # "right" is contested


def test_fully_trapped_still_returns_a_move():
    me = make_snake("me", [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)])
    state = make_state(me)

    assert choose_move(state) in DIRECTIONS


def test_choice_matches_independent_model_recomputation():
    # Two candidates (up/right) from the left wall; recompute the standardized
    # linear score by hand and expect the same argmax.
    me = make_snake("me", [(0, 5), (0, 4), (0, 3)], health=30)
    state = make_state(me, food=[(4, 5)])

    def score(move):
        feats = candidate_features(state, move)
        total = MODEL["intercept"]
        for i, name in enumerate(MODEL["feature_names"]):
            std = MODEL["std"][i]
            z = (feats.get(name, 0.0) - MODEL["mean"][i]) / std if std else 0.0
            total += MODEL["coef"][i] * z
        return total

    expected = max(["up", "right"], key=score)
    assert choose_move(state) == expected


def test_logic_choose_move_falls_back_to_heuristic(monkeypatch):
    from logic import choose_move as logic_choose_move

    def boom(_state):
        raise RuntimeError("model broke")

    monkeypatch.setattr(policy, "choose_move", boom)

    me = make_snake("me", [(5, 5), (5, 4), (5, 3)])
    state = make_state(me, food=[(0, 0)])

    assert logic_choose_move(state) in DIRECTIONS


def test_logic_choose_move_uses_policy_pipeline():
    from logic import choose_move as logic_choose_move

    me = make_snake("me", [(5, 5), (5, 4), (5, 3)])
    enemy = make_snake("e", [(7, 5), (8, 5), (9, 5), (9, 4)])
    state = make_state(me, snakes=[me, enemy])

    assert logic_choose_move(state) in {"up", "left"}
