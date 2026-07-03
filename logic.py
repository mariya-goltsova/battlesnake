"""Move-selection entry point for the Battlesnake server.

The served policy is the hybrid pipeline in policy.py (hard safety rules,
then model ranking of the safe candidates). A compact heuristic remains as
a fallback so gameplay still returns a legal move if the pipeline fails.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

from typing import Dict

import policy
from core import (
    DIRECTIONS,
    Point,
    flood_fill,
    food_points,
    head_to_head_cells,
    in_bounds,
    manhattan,
    occupied_cells,
)

# Penalty applied to a move that could lose a head-to-head collision.
HEAD_TO_HEAD_PENALTY = 10_000
# Below this health we start actively steering toward food.
HUNGRY_THRESHOLD = 50


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.2.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using the hybrid pipeline, with a heuristic fallback."""
    try:
        return policy.choose_move(game_state)
    except Exception:  # noqa: BLE001 - a pipeline issue must never break gameplay
        return choose_move_heuristic(game_state)


def choose_move_heuristic(game_state: Dict) -> str:
    """Greedy one-step heuristic kept as the last-resort fallback."""
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    occupied = occupied_cells(board["snakes"])
    danger = head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = food_points(board)

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        # Reachable open space from this cell. If we can't fit our own body in
        # the space we'd be moving into, we're about to trap ourselves.
        space = flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        # When hungry, nudge toward the closest food.
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score = score
            best_move = move

    # No safe move found -> we're cornered. Move up and hope for the best.
    return best_move or "up"
