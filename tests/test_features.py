"""Tests for features.py — shared feature extraction on the simulated map."""

from helpers import make_snake, make_state

from features import MODEL_FEATURE_NAMES, candidate_features


def test_food_behind_body_wall_is_unreachable():
    # Food at (1,3) is manhattan-close but walled off by an enemy column at
    # x=3 (stacked tail, so no gap opens). BFS must see it as unreachable.
    wall = make_snake(
        "w", [(3, 6), (3, 5), (3, 4), (3, 3), (3, 2), (3, 1), (3, 0), (3, 0)]
    )
    me = make_snake("me", [(5, 3), (5, 2), (5, 1)], health=30)  # hungry
    state = make_state(me, snakes=[me, wall], food=[(1, 3)], width=7, height=7)

    feats = candidate_features(state, "left")  # next: (4, 3)

    assert feats["food_reachable"] == 0.0
    assert feats["food_score"] == 0.0
    assert feats["bfs_food_dist"] >= 10_000


def test_food_inside_ring_opens_when_own_tail_frees():
    # Our body is a closed ring around (3,3); moving onto our vacating tail
    # (2,3) puts the enclosed food one step away on the simulated map.
    me = make_snake(
        "me",
        [(2, 2), (3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (2, 4), (2, 3)],
        health=30,
    )
    state = make_state(me, food=[(3, 3)], width=7, height=7)

    feats = candidate_features(state, "up")  # onto the tail cell (2,3)

    assert feats["bfs_food_dist"] == 1.0
    assert feats["food_reachable"] == 1.0
    assert feats["is_food"] == 0.0
    assert feats["reaches_tail"] == 1.0  # new tail (2,4) is adjacent


def test_moving_onto_food_sets_is_food_and_zero_distance():
    me = make_snake("me", [(5, 5), (5, 4), (5, 3)], health=30)
    state = make_state(me, food=[(5, 6)])

    feats = candidate_features(state, "up")

    assert feats["is_food"] == 1.0
    assert feats["bfs_food_dist"] == 0.0
    assert feats["food_reachable"] == 1.0


def test_space_counted_on_simulated_map():
    # A straight snake in the open: after moving, the freed tail cell counts
    # as open space, so open_space == all remaining free cells.
    me = make_snake("me", [(2, 2), (2, 1), (2, 0)])
    state = make_state(me, width=5, height=5)

    feats = candidate_features(state, "up")  # next (2,3); body (2,2),(2,1); (2,0) freed

    # 25 cells minus the 2 body cells still blocking; the flood counts its own
    # start cell (original feature semantics) and the freed tail (2,0) is open.
    assert feats["open_space"] == 23.0


def test_all_model_feature_names_present_and_float():
    me = make_snake("me", [(5, 5), (5, 4), (5, 3)], health=80)
    enemy = make_snake("e", [(8, 8), (8, 7), (8, 6)])
    state = make_state(me, snakes=[me, enemy], food=[(0, 0)])

    feats = candidate_features(state, "up")

    assert set(MODEL_FEATURE_NAMES) <= set(feats)
    assert len(MODEL_FEATURE_NAMES) == 13
    assert all(isinstance(feats[name], float) for name in feats)
    # not hungry (health 80) -> food_score stays 0 by design
    assert feats["food_score"] == 0.0
