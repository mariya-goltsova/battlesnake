"""Tests for features.py (computed on the simulated post-move state) and for
the simulate_after_move occupancy model it builds on."""

from src_rl.features import FEATURE_NAMES, UNREACHABLE, candidate_features
from src_rl.sim import simulate_after_move
from src_rl.tests.helpers import make_snake, make_state


# --- simulate_after_move occupancy ------------------------------------------------------


def test_own_tail_vacates_when_not_eating():
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)])
    sim = simulate_after_move(make_state(me), "down")
    assert sim["next_pos"] == (5, 4) and not sim["ate"]
    assert sim["my_body"] == [(5, 4), (5, 5), (5, 6)]
    assert (5, 7) not in sim["occupied"]


def test_own_tail_kept_when_eating():
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)])
    sim = simulate_after_move(make_state(me, food=[(5, 4)]), "down")
    assert sim["ate"]
    assert sim["my_body"] == [(5, 4), (5, 5), (5, 6), (5, 7)]
    assert (5, 7) in sim["occupied"]


def test_enemy_tail_vacates_unless_it_may_eat():
    me = make_snake("me", [(0, 0), (0, 1)])
    enemy = make_snake("e", [(7, 5), (6, 5), (5, 5)])
    assert (5, 5) not in simulate_after_move(make_state(me, snakes=[me, enemy]), "right")["occupied"]
    withfood = make_state(me, snakes=[me, enemy], food=[(8, 5)])
    assert (5, 5) in simulate_after_move(withfood, "right")["occupied"]


def test_enemy_stacked_tail_stays_occupied():
    me = make_snake("me", [(0, 0), (0, 1)])
    enemy = make_snake("e", [(7, 5), (6, 5), (5, 5), (5, 5)])  # ate last tick
    sim = simulate_after_move(make_state(me, snakes=[me, enemy]), "right")
    assert (5, 5) in sim["occupied"]


# --- feature values on a hand-computable solo board -------------------------------------


def test_features_solo_board_exact_values():
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)], health=100)
    state = make_state(me, food=[(0, 0)])
    feats = candidate_features(state, "down")  # nxt = (5,4); tail (5,7) vacates

    assert feats["space_capped"] == 4.0
    assert feats["open_space"] == 118.0  # 121 - 3 occupied (post-move body)
    assert feats["voronoi"] == 118.0
    assert feats["reaches_tail"] == 1.0
    assert feats["escape"] == 3.0
    assert feats["h2h_danger"] == 0.0
    assert feats["near_bigger_head"] == 22.0
    assert feats["near_enemy_head"] == 22.0
    assert feats["wall_dist"] == 4.0
    assert feats["food_score"] == 0.0  # not hungry
    assert feats["food_delta"] == 1.0  # BFS 10 -> 9
    assert feats["is_food"] == 0.0
    assert feats["dist_to_center"] == 1.0
    assert feats["bfs_food_dist"] == 9.0
    assert feats["food_reachable"] == 1.0
    assert feats["length_advantage"] == 3.0  # no enemies

    assert set(feats) == set(FEATURE_NAMES)


def test_food_behind_wall_is_unreachable_by_bfs():
    # Enemy body walls off the bottom-right pocket that holds the food. The fix
    # over legacy: BFS sees the wall (legacy's Manhattan pretended dist 2).
    me = make_snake("me", [(0, 5), (0, 6), (0, 7)], health=30)
    # Stacked tail (just ate) so the row stays solid — otherwise the vacating
    # tail correctly opens a path around the wall.
    wall = make_snake(
        "w",
        [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4), (10, 4)],
    )
    state = make_state(me, snakes=[me, wall], food=[(0, 2)])
    feats = candidate_features(state, "right")  # nxt (1,5); food at (0,2) behind wall
    assert feats["food_reachable"] == 0.0
    assert feats["bfs_food_dist"] == UNREACHABLE
    assert feats["food_score"] == 0.0  # hungry but food unreachable -> no pull


def test_open_space_sees_tail_corridor():
    # A corridor sealed by our own tail: legacy counts the tail as a permanent
    # wall; the simulated state frees it, opening the far side.
    me = make_snake("me", [(1, 1), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])
    state = make_state(me, width=7, height=3)
    feats = candidate_features(state, "right")  # nxt (2,1); tail (5,0) vacates
    assert feats["open_space"] >= 15.0  # 21 cells - post-move body(6) = 15 reachable


def test_h2h_uses_post_feed_length():
    # We eat this move (len 3 -> 4); enemy len 3 becomes shorter than us, so
    # its head-adjacent cells are no longer losing-h2h cells.
    me = make_snake("me", [(4, 5), (4, 6), (4, 7)])
    enemy = make_snake("e", [(6, 5), (7, 5), (8, 5)])
    state = make_state(me, snakes=[me, enemy], food=[(5, 5)])
    feats = candidate_features(state, "right")  # nxt (5,5): food + adjacent to e
    assert feats["is_food"] == 1.0
    assert feats["h2h_danger"] == 0.0  # post-feed we are longer
    assert feats["length_advantage"] == 1.0  # 4 vs 3


def test_length_advantage_negative_when_outgrown():
    me = make_snake("me", [(4, 5), (4, 6)])
    enemy = make_snake("e", [(9, 9), (9, 8), (9, 7), (9, 6)])
    state = make_state(me, snakes=[me, enemy])
    feats = candidate_features(state, "down")
    assert feats["length_advantage"] == -2.0  # 2 vs 4
