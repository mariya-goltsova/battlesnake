"""Fixed-board tests for safety.py: tail-aware legality, hard vetoes, fallback.

Veto ordering matters: certain death (trap with no follow-up) is filtered
before probable death (losing head-to-head), and each tier is applied only if
at least one move survives it — so the filter never empties a non-empty input.
"""

from src_rl.safety import hard_safety_filter, least_bad_move, legal_moves
from src_rl.tests.helpers import make_snake, make_state


# --- legal_moves: tail-aware occupancy ------------------------------------------------


def test_own_tail_chase_is_legal():
    # Curled in the corner: head (0,0), tail (0,1). Moving up onto the tail is
    # legal because the tail vacates this same tick (legacy wrongly forbids it).
    me = make_snake("me", [(0, 0), (1, 0), (1, 1), (0, 1)])
    state = make_state(me)
    assert "up" in legal_moves(state)


def test_own_stacked_tail_is_blocked():
    # Just ate: body[-1] == body[-2], the tail cell does not vacate this tick.
    me = make_snake("me", [(0, 0), (1, 0), (1, 1), (0, 1), (0, 1)])
    state = make_state(me)
    assert "up" not in legal_moves(state)


def test_enemy_tail_legal_when_no_food_near_enemy_head():
    me = make_snake("me", [(4, 5), (4, 4), (4, 3)])
    enemy = make_snake("e", [(7, 5), (6, 5), (5, 5)])  # tail at (5,5), head far
    state = make_state(me, snakes=[me, enemy])
    assert "right" in legal_moves(state)  # (5,5) vacates


def test_enemy_tail_blocked_when_enemy_may_eat():
    me = make_snake("me", [(4, 5), (4, 4), (4, 3)])
    enemy = make_snake("e", [(7, 5), (6, 5), (5, 5)])
    state = make_state(me, snakes=[me, enemy], food=[(8, 5)])  # food next to e's head
    assert "right" not in legal_moves(state)


def test_walls_and_bodies_still_blocked():
    me = make_snake("me", [(0, 1), (0, 2), (0, 3)])
    enemy = make_snake("e", [(1, 1), (2, 1), (3, 1)])
    state = make_state(me, snakes=[me, enemy])
    # up = own neck, left = wall, right = enemy head cell (their neck next tick)
    assert legal_moves(state) == ["down"]


# --- hard_safety_filter: head-to-head veto ---------------------------------------------


def _h2h_board(enemy_len):
    # me at (5,5); enemy head at (7,5): its h2h cells include (6,5) = our "right".
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)])
    enemy_body = [(7, 5), (8, 5), (9, 5), (9, 6)][:enemy_len]
    enemy = make_snake("e", enemy_body)
    return make_state(me, snakes=[me, enemy])


def test_filter_vetoes_losing_h2h_when_alternative_exists():
    state = _h2h_board(enemy_len=4)  # longer enemy
    legal = legal_moves(state)
    assert "right" in legal
    safe = hard_safety_filter(state, legal)
    assert "right" not in safe and safe


def test_filter_vetoes_tying_h2h_equal_length():
    state = _h2h_board(enemy_len=3)  # equal length -> tie kills both
    safe = hard_safety_filter(state, legal_moves(state))
    assert "right" not in safe and safe


def test_filter_allows_h2h_versus_shorter():
    state = _h2h_board(enemy_len=2)  # we win this collision
    assert "right" in hard_safety_filter(state, legal_moves(state))


def test_filter_keeps_h2h_move_when_it_is_the_only_move():
    # Head in the corner, own body above: the only legal move is an h2h cell of
    # an equal-length enemy -> the veto must not empty the list.
    me = make_snake("me", [(0, 0), (0, 1), (0, 2), (0, 3)])
    enemy = make_snake("e", [(2, 0), (2, 1), (2, 2), (3, 2)])
    state = make_state(me, snakes=[me, enemy])
    legal = legal_moves(state)
    assert legal == ["right"]  # (1,0), adjacent to enemy head
    assert hard_safety_filter(state, legal) == ["right"]


def test_filter_vetoes_immediate_trap_when_alternative_exists():
    # Moving left enters a 1-cell dead-end pocket (no legal follow-up move);
    # moving right stays in open space.
    me = make_snake("me", [(1, 0), (1, 1), (1, 2), (2, 2)])
    wall = make_snake("w", [(0, 1), (0, 2), (0, 3)])
    state = make_state(me, snakes=[me, wall])
    legal = legal_moves(state)
    assert set(legal) == {"left", "right"}
    assert hard_safety_filter(state, legal) == ["right"]


def test_certain_death_vetoed_before_probable_death():
    # left = certain death (trap), right = probable death (h2h cell of a longer
    # enemy). The filter must keep the merely-probable death, not the certain one.
    me = make_snake("me", [(1, 0), (1, 1), (1, 2), (2, 2)])
    wall = make_snake("w", [(0, 1), (0, 2), (0, 3)])
    enemy = make_snake("e", [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1)])
    state = make_state(me, snakes=[me, wall, enemy])
    legal = legal_moves(state)
    assert set(legal) == {"left", "right"}
    assert hard_safety_filter(state, legal) == ["right"]


# --- least_bad_move ---------------------------------------------------------------------


def test_least_bad_returns_a_direction_even_when_fully_cornered():
    me = make_snake("me", [(0, 0), (1, 0), (1, 1), (0, 1), (0, 1)])
    state = make_state(me, width=2, height=2)
    assert legal_moves(state) == []
    assert least_bad_move(state) in {"up", "down", "left", "right"}


def test_least_bad_prefers_maybe_vacating_tail_over_certain_death():
    # No legal moves: up = own body, left/down = walls, right = enemy tail that
    # is conservatively blocked (food near enemy head) but MIGHT still vacate.
    # The only non-zero-chance move is right.
    me = make_snake("me", [(0, 0), (0, 1), (0, 2)])
    enemy = make_snake("e", [(3, 0), (2, 0), (1, 0)])  # tail at (1,0)
    state = make_state(me, snakes=[me, enemy], food=[(4, 0)])
    assert legal_moves(state) == []
    assert least_bad_move(state) == "right"
