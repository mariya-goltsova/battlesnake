"""Tests for sim.py — next-state simulation with correct tail handling."""

from helpers import make_snake, make_state

from sim import blocked_for_legality, simulate_after_move


# --- simulate_after_move: own snake -------------------------------------------


def test_move_without_eating_frees_tail_and_shifts_body():
    me = make_snake("me", [(5, 5), (5, 4), (5, 3)])
    state = make_state(me)

    res = simulate_after_move(state, "up")

    assert res["next_pos"] == (5, 6)
    assert res["ate"] is False
    assert res["my_body"] == [(5, 6), (5, 5), (5, 4)]  # tail (5,3) dropped
    assert (5, 3) not in res["occupied"]
    assert {(5, 6), (5, 5), (5, 4)} <= res["occupied"]


def test_move_onto_food_grows_and_keeps_tail():
    me = make_snake("me", [(5, 5), (5, 4), (5, 3)])
    state = make_state(me, food=[(5, 6)])

    res = simulate_after_move(state, "up")

    assert res["ate"] is True
    assert res["my_body"] == [(5, 6), (5, 5), (5, 4), (5, 3)]  # length + 1
    assert (5, 3) in res["occupied"]


def test_stacked_tail_stays_occupied_when_not_eating():
    # body[-1] == body[-2]: we just ate, so the duplicated tail cell does not
    # free even though this move eats nothing.
    me = make_snake("me", [(5, 5), (5, 4), (5, 3), (5, 3)])
    state = make_state(me)

    res = simulate_after_move(state, "up")

    assert res["my_body"] == [(5, 6), (5, 5), (5, 4), (5, 3)]
    assert (5, 3) in res["occupied"]


# --- simulate_after_move: enemy tails ------------------------------------------


def test_enemy_tail_freed_when_enemy_cannot_eat():
    me = make_snake("me", [(1, 1), (1, 0)])
    enemy = make_snake("e", [(8, 8), (8, 7), (8, 6)])
    state = make_state(me, snakes=[me, enemy])  # no food anywhere

    res = simulate_after_move(state, "up")

    assert (8, 6) not in res["occupied"]  # enemy tail freed
    assert {(8, 8), (8, 7)} <= res["occupied"]


def test_enemy_tail_kept_when_food_adjacent_to_enemy_head():
    me = make_snake("me", [(1, 1), (1, 0)])
    enemy = make_snake("e", [(8, 8), (8, 7), (8, 6)])
    state = make_state(me, snakes=[me, enemy], food=[(8, 9)])  # enemy may eat

    res = simulate_after_move(state, "up")

    assert (8, 6) in res["occupied"]  # conservative: tail stays


def test_enemy_stacked_tail_kept_even_without_food():
    me = make_snake("me", [(1, 1), (1, 0)])
    enemy = make_snake("e", [(8, 8), (8, 7), (8, 6), (8, 6)])
    state = make_state(me, snakes=[me, enemy])

    res = simulate_after_move(state, "up")

    assert (8, 6) in res["occupied"]  # duplicated segment still there


# --- blocked_for_legality --------------------------------------------------------


def test_own_tail_cell_is_legal_to_enter():
    # Snake curled so its tail is adjacent to its head: 3x2 ring.
    me = make_snake("me", [(2, 2), (3, 2), (3, 3), (2, 3)])
    state = make_state(me)

    blocked = blocked_for_legality(state)

    assert (2, 3) not in blocked  # tail about to move away
    assert {(2, 2), (3, 2), (3, 3)} <= blocked


def test_own_stacked_tail_cell_is_blocked():
    me = make_snake("me", [(2, 2), (3, 2), (3, 3), (2, 3), (2, 3)])
    state = make_state(me)

    blocked = blocked_for_legality(state)

    assert (2, 3) in blocked  # stacked tail does not free this tick


def test_enemy_tail_legality_respects_adjacent_food():
    me = make_snake("me", [(1, 1), (1, 0)])
    enemy = make_snake("e", [(8, 8), (8, 7), (8, 6)])

    no_food = make_state(me, snakes=[me, enemy])
    assert (8, 6) not in blocked_for_legality(no_food)

    food_near_enemy = make_state(me, snakes=[me, enemy], food=[(7, 8)])
    assert (8, 6) in blocked_for_legality(food_near_enemy)
