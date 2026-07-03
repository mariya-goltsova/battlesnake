"""Characterization tests pinning the behavior of the frozen legacy policy.

These document what the deployed baseline (commit 7507bea) actually does —
including its known quirks (tails treated as permanently occupied, Manhattan
food distance). They must keep passing forever: legacy_policy.py is frozen.
"""

from src_rl.baselines import legacy_policy as legacy
from src_rl.tests.helpers import make_snake, make_state


# --- primitives -----------------------------------------------------------------


def test_flood_fill_counts_pocket_and_respects_cap():
    walls = {(2, 0), (2, 1), (0, 2), (1, 2), (2, 2)}
    assert legacy._flood_fill((0, 0), walls, 5, 5, limit=100) == 4
    assert legacy._flood_fill((0, 0), set(), 11, 11, limit=5) == 5


def test_bfs_dist_seeds_sources_even_if_blocked_and_walls_split():
    dist = legacy._bfs_dist([(1, 1)], {(1, 1)}, 3, 3)
    assert dist[(1, 1)] == 0
    wall = {(2, y) for y in range(5)}
    dist = legacy._bfs_dist([(0, 0)], wall, 5, 5)
    assert (1, 4) in dist and (3, 0) not in dist


def test_legal_moves_treat_tails_as_blocked():
    # Known pessimistic quirk: the tail cell (5,7) frees up next turn unless the
    # snake eats, but legacy counts it as occupied.
    me = make_snake("me", [(5, 6), (5, 7)])
    other = make_snake("b", [(5, 5), (4, 5)])
    state = make_state(me, snakes=[me, other])
    legal = legacy._legal_moves(state)
    assert "up" not in legal  # own tail at (5,7)
    assert "down" not in legal  # enemy head at (5,5)
    assert set(legal) == {"left", "right"}


# --- feature vector on a hand-computable solo board -------------------------------


def test_candidate_features_solo_board_exact_values():
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)], health=100)
    state = make_state(me, food=[(0, 0)])
    feats = legacy._candidate_features(state, "down")  # nxt = (5, 4)

    assert feats["space_capped"] == 4.0  # capped at my_length + 1
    assert feats["open_space"] == 118.0  # 121 cells - 3 body segments
    assert feats["voronoi"] == 118.0  # no enemies -> every reachable cell is ours
    assert feats["reaches_tail"] == 1.0
    assert feats["escape"] == 3.0  # (5,5) is our neck
    assert feats["h2h_danger"] == 0.0
    assert feats["near_bigger_head"] == 22.0  # width + height default
    assert feats["near_enemy_head"] == 22.0
    assert feats["wall_dist"] == 4.0
    assert feats["food_score"] == 0.0  # health 100 -> not hungry
    assert feats["food_delta"] == 1.0  # manhattan 10 -> 9
    assert feats["is_food"] == 0.0
    assert feats["dist_to_center"] == 1.0

    assert set(feats) == set(legacy._MODEL["feature_names"])


def test_candidate_features_manhattan_food_quirk():
    # Known quirk: food distance ignores bodies. A wall of enemy body between us
    # and the food does not change food_delta.
    me = make_snake("me", [(0, 5), (0, 6)], health=30)  # hungry
    wall = make_snake("w", [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)])
    state = make_state(me, snakes=[me, wall], food=[(2, 0)])
    feats = legacy._candidate_features(state, "down")  # nxt (0,4), food behind wall
    assert feats["food_delta"] == 1.0  # pretends food got closer


# --- move selection ----------------------------------------------------------------


def test_choose_move_returns_legal_and_deterministic():
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)])
    enemy = make_snake("e", [(2, 2), (2, 3), (2, 4), (2, 5)])
    state = make_state(me, snakes=[me, enemy], food=[(9, 9)], turn=7)
    move = legacy.choose_move(state)
    assert move in legacy._legal_moves(state)
    assert all(legacy.choose_move(state) == move for _ in range(3))


def test_choose_move_heuristic_cornered_returns_up():
    # Fully cornered snake: heuristic answers "up" (and dies) — pinned behavior.
    me = make_snake("me", [(0, 0), (1, 0), (1, 1), (0, 1)])
    state = make_state(me, width=2, height=2)
    assert legacy.choose_move_heuristic(state) == "up"
