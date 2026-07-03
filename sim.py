"""Next-state simulation: board occupancy after a candidate move.

Fixes the static-occupancy problem of the base bot: a tail cell frees on the
next tick unless the snake grows, and a snake that just ate carries a stacked
tail (``body[-1] == body[-2]``) that does NOT free.

Enemy movement is not predicted here — enemy bodies stay in place except for
tail handling; the cells an enemy head may enter are the safety filter's job.
"""

from typing import Dict, List, Set

from core import DIRECTIONS, Point, food_points, snake_body


def _tail_frees(body: List[Point]) -> bool:
    """True if the tail cell vacates when this snake moves without eating."""
    return len(body) >= 2 and body[-1] != body[-2]


def _may_eat(head: Point, foods: Set[Point]) -> bool:
    """True if any food is adjacent to ``head`` (the snake could eat this tick)."""
    return any((head[0] + dx, head[1] + dy) in foods for dx, dy in DIRECTIONS.values())


def _enemy_cells(snake: Dict, foods: Set[Point]) -> Set[Point]:
    """An enemy's occupied cells for the next tick (conservative tail handling).

    The tail frees only when it is not stacked AND the enemy cannot eat this
    tick; if it *could* eat, we assume it does and keep the tail occupied.
    """
    body = snake_body(snake)
    cells = set(body)
    if _tail_frees(body) and not _may_eat(body[0], foods):
        cells.discard(body[-1])
    return cells


def simulate_after_move(state: Dict, move: str) -> Dict:
    """Simulate our snake playing ``move`` (assumed legal).

    Returns ``{"next_pos": Point, "ate": bool, "my_body": List[Point],
    "occupied": Set[Point]}`` where ``occupied`` is the full board occupancy
    after our move resolves (our new body plus enemy cells).
    """
    board = state["board"]
    you = state["you"]
    body = snake_body(you)
    head = body[0]
    foods = set(food_points(board))

    dx, dy = DIRECTIONS[move]
    next_pos = (head[0] + dx, head[1] + dy)
    ate = next_pos in foods

    # Growth keeps the tail; a normal move drops it.
    my_body = [next_pos] + (body if ate else body[:-1])

    occupied: Set[Point] = set(my_body)
    for snake in board["snakes"]:
        if snake["id"] == you["id"]:
            continue
        occupied |= _enemy_cells(snake, foods)

    return {"next_pos": next_pos, "ate": ate, "my_body": my_body, "occupied": occupied}


def blocked_for_legality(state: Dict) -> Set[Point]:
    """Cells we may not move into on the current tick.

    All snake segments, minus tails that are about to free: our own tail frees
    unless stacked (stepping onto our tail can never feed us — food is never
    under a body), an enemy tail frees unless stacked or the enemy could eat.
    """
    board = state["board"]
    you = state["you"]
    foods = set(food_points(board))

    blocked: Set[Point] = set()
    for snake in board["snakes"]:
        if snake["id"] == you["id"]:
            body = snake_body(snake)
            cells = set(body)
            if _tail_frees(body):
                cells.discard(body[-1])
            blocked |= cells
        else:
            blocked |= _enemy_cells(snake, foods)
    return blocked
