"""Hard safety rules: legal moves and the head-to-head veto.

This is the rules layer of the hybrid pipeline. It removes moves that are
certain or near-certain deaths (walls, bodies, losing head-to-heads) and
leaves ranking the surviving candidates to the scorer. When everything is
dangerous it degrades gracefully instead of returning nothing useful:
head-to-head cells are only vetoed while an alternative exists.
"""

from typing import Dict, List

from core import DIRECTIONS, Point, head_to_head_cells, in_bounds, snake_body
from sim import blocked_for_legality


def _next_cell(state: Dict, move: str) -> Point:
    head = snake_body(state["you"])[0]
    dx, dy = DIRECTIONS[move]
    return (head[0] + dx, head[1] + dy)


def legal_moves(state: Dict) -> List[str]:
    """Moves that stay on the board and don't enter a blocked cell.

    Tail-aware: a cell whose tail is about to vacate counts as free
    (see sim.blocked_for_legality).
    """
    board = state["board"]
    width, height = board["width"], board["height"]
    blocked = blocked_for_legality(state)
    return [
        move
        for move in DIRECTIONS
        if in_bounds(_next_cell(state, move), width, height)
        and _next_cell(state, move) not in blocked
    ]


def hard_safety_filter(state: Dict) -> List[str]:
    """Legal moves minus losing head-to-head cells, while alternatives exist.

    A candidate cell adjacent to the head of an enemy with ``length >=`` ours
    is dropped only if at least one non-contested legal move remains. If ALL
    legal moves are contested, they are all returned, ordered so that cells
    contested only by equal-length enemies (a tie) come before cells contested
    by strictly longer ones (a loss). Returns ``[]`` when fully trapped —
    the policy layer owns the least-deadly fallback.
    """
    you = state["you"]
    snakes = state["board"]["snakes"]
    my_length = you["length"]

    legal = legal_moves(state)
    contested = head_to_head_cells(snakes, you["id"], my_length)
    safe = [move for move in legal if _next_cell(state, move) not in contested]
    if safe:
        return safe

    # Everything is contested: prefer a possible tie over a guaranteed loss.
    losing = head_to_head_cells(snakes, you["id"], my_length + 1)  # strictly longer
    return sorted(legal, key=lambda move: 1 if _next_cell(state, move) in losing else 0)
