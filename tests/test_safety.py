"""Tests for safety.py — legal moves and the hard safety filter (h2h veto).

Covers the canonical board states from the plan: wall, own tail, stacked
tail, h2h vs longer/shorter, all-moves-contested, dead end (layer boundary)
and fully trapped.
"""

from helpers import make_snake, make_state

from safety import hard_safety_filter, legal_moves


# 1. wall: head in a corner — only in-bounds, non-body moves are legal.
def test_corner_wall_leaves_single_legal_move():
    me = make_snake("me", [(0, 0), (0, 1), (0, 2)])
    state = make_state(me)
    assert legal_moves(state) == ["right"]


# 2. own tail: moving onto our (non-stacked) tail is legal.
def test_own_tail_is_legal():
    me = make_snake("me", [(2, 2), (3, 2), (3, 3), (2, 3)])
    state = make_state(me)
    assert "up" in legal_moves(state)  # (2, 3) is our vacating tail


# 3. stacked tail: the same cell is illegal right after eating.
def test_own_stacked_tail_is_illegal():
    me = make_snake("me", [(2, 2), (3, 2), (3, 3), (2, 3), (2, 3)])
    state = make_state(me)
    assert "up" not in legal_moves(state)


# 4. h2h vs longer enemy: contested cell dropped while an alternative exists.
def test_h2h_cell_vetoed_against_longer_enemy():
    me = make_snake("me", [(5, 5), (5, 4), (5, 3)])
    enemy = make_snake("e", [(7, 5), (8, 5), (9, 5), (9, 4)])  # length 4 > 3
    state = make_state(me, snakes=[me, enemy])

    result = hard_safety_filter(state)

    assert "right" not in result  # (6,5) is adjacent to the longer enemy's head
    assert set(result) == {"up", "left"}


# 5. h2h vs shorter enemy: contested cell is NOT dropped.
def test_h2h_cell_kept_against_shorter_enemy():
    me = make_snake("me", [(5, 5), (5, 4), (5, 3)])
    enemy = make_snake("e", [(7, 5), (8, 5)])  # length 2 < 3
    state = make_state(me, snakes=[me, enemy])

    assert "right" in hard_safety_filter(state)


# 6. all moves contested: nothing is dropped; ties (equal length) come first.
def test_all_contested_returns_all_with_equal_length_first():
    me = make_snake("me", [(5, 5), (5, 4), (5, 3)])  # length 3
    equal = make_snake("eq", [(5, 7), (5, 8), (5, 9)])  # contests up (5,6)
    longer_l = make_snake("L1", [(3, 5), (2, 5), (1, 5), (1, 4)])  # contests left (4,5)
    longer_r = make_snake("L2", [(7, 5), (8, 5), (9, 5), (9, 6)])  # contests right (6,5)
    state = make_state(me, snakes=[me, equal, longer_l, longer_r])

    result = hard_safety_filter(state)

    assert set(result) == {"up", "left", "right"}
    assert result[0] == "up"  # tie beats a guaranteed loss


# 7. dead end vs open: both stay — space evaluation is scoring's job.
def test_dead_end_not_filtered_by_safety_layer():
    me = make_snake("me", [(1, 0), (1, 1), (0, 1)])
    state = make_state(me)
    assert set(hard_safety_filter(state)) == {"left", "right"}


# 8. fully trapped: empty list (policy handles the fallback).
def test_fully_trapped_returns_empty():
    me = make_snake("me", [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)])
    state = make_state(me)
    assert legal_moves(state) == []
    assert hard_safety_filter(state) == []
