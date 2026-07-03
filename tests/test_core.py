"""Tests for core.py — shared board primitives extracted from logic.py."""

from core import (
    DIRECTIONS,
    bfs_dist,
    flood_fill,
    food_points,
    head_to_head_cells,
    in_bounds,
    manhattan,
    occupied_cells,
    snake_body,
)


def make_snake(snake_id, body, health=100):
    """Build an API-schema snake dict from a list of (x, y) points."""
    return {
        "id": snake_id,
        "health": health,
        "length": len(body),
        "head": {"x": body[0][0], "y": body[0][1]},
        "body": [{"x": x, "y": y} for x, y in body],
    }


# --- directions / geometry ----------------------------------------------------


def test_directions_map_matches_coordinate_system():
    assert DIRECTIONS == {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0),
    }


def test_in_bounds_edges():
    assert in_bounds((0, 0), 5, 5)
    assert in_bounds((4, 4), 5, 5)
    assert not in_bounds((-1, 0), 5, 5)
    assert not in_bounds((0, -1), 5, 5)
    assert not in_bounds((5, 0), 5, 5)
    assert not in_bounds((0, 5), 5, 5)


def test_manhattan_symmetry():
    assert manhattan((0, 0), (3, 4)) == 7
    assert manhattan((3, 4), (0, 0)) == 7
    assert manhattan((2, 2), (2, 2)) == 0


# --- bfs_dist -------------------------------------------------------------------


def test_bfs_dist_empty_board():
    dist = bfs_dist([(0, 0)], set(), 5, 5)
    assert dist[(0, 0)] == 0
    assert dist[(4, 4)] == 8
    assert len(dist) == 25


def test_bfs_dist_wall_splits_board():
    # Vertical wall at x=2 splits a 5x5 board; right side unreachable from (0,0).
    wall = {(2, y) for y in range(5)}
    dist = bfs_dist([(0, 0)], wall, 5, 5)
    assert (1, 4) in dist
    assert (3, 0) not in dist
    assert (4, 4) not in dist


def test_bfs_dist_source_seeded_even_if_blocked():
    # Sources are seeded at distance 0 even when listed in blocked.
    dist = bfs_dist([(1, 1)], {(1, 1)}, 3, 3)
    assert dist[(1, 1)] == 0


# --- flood_fill -----------------------------------------------------------------


def test_flood_fill_uncapped_counts_pocket():
    # 2x2 pocket in the corner, walled off from the rest of a 5x5 board.
    walls = {(2, 0), (2, 1), (0, 2), (1, 2), (2, 2)}
    assert flood_fill((0, 0), walls, 5, 5) == 4


def test_flood_fill_capped_stops_at_limit():
    assert flood_fill((0, 0), set(), 11, 11, limit=5) == 5


# --- snake helpers ----------------------------------------------------------------


def test_snake_body_and_food_points():
    snake = make_snake("s1", [(1, 1), (1, 2), (1, 3)])
    assert snake_body(snake) == [(1, 1), (1, 2), (1, 3)]
    board = {"food": [{"x": 4, "y": 4}, {"x": 0, "y": 0}]}
    assert food_points(board) == [(4, 4), (0, 0)]


def test_occupied_cells_includes_all_segments_and_tails():
    snakes = [
        make_snake("a", [(1, 1), (1, 2), (1, 3)]),
        make_snake("b", [(3, 3), (3, 4)]),
    ]
    assert occupied_cells(snakes) == {(1, 1), (1, 2), (1, 3), (3, 3), (3, 4)}


def test_head_to_head_cells_only_from_equal_or_longer_enemies():
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)])  # length 3
    shorter = make_snake("short", [(0, 0), (0, 1)])  # length 2 -> no danger
    equal = make_snake("eq", [(9, 9), (9, 8), (9, 7)])  # length 3 -> danger
    snakes = [me, shorter, equal]

    danger = head_to_head_cells(snakes, "me", 3)
    assert (0, 1) not in danger and (1, 0) not in danger
    assert danger == {(9, 10), (9, 8), (8, 9), (10, 9)}


def test_head_to_head_cells_ignores_own_head():
    me = make_snake("me", [(5, 5), (5, 6), (5, 7)])
    danger = head_to_head_cells([me], "me", 3)
    assert danger == set()


# --- regression: logic.py still works after the refactor --------------------------


def test_logic_choose_move_still_returns_legal_move():
    from logic import choose_move

    me = make_snake("me", [(5, 5), (5, 6), (5, 7)])
    state = {
        "turn": 1,
        "board": {
            "width": 11,
            "height": 11,
            "food": [{"x": 0, "y": 0}],
            "snakes": [me],
        },
        "you": me,
    }
    move = choose_move(state)
    assert move in DIRECTIONS
    # (5,6) is our neck — the only illegal direction from (5,5) is "up".
    assert move != "up"
