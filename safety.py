"""Hard safety layer: tail-aware legality, certain/probable-death vetoes, fallback.

Veto ordering: certain death (no legal follow-up) is filtered before probable
death (losing/tying head-to-head), and each tier applies only if at least one
move survives it — the filter never empties a non-empty input. When nothing is
legal at all, ``least_bad_move`` picks the direction with any chance of
survival instead of legacy's "up and hope".
"""

from typing import Dict, List, Optional, Set

from src_rl.core import DIRECTIONS, Point, head_to_head_cells, in_bounds, snake_body
from src_rl.sim import simulate_after_move


def _blocked_cells(game_state: Dict, optimistic: bool = False) -> Set[Point]:
    """Cells we cannot enter this tick.

    A tail cell is free unless it is stacked (its snake ate last tick) or —
    conservatively — its snake may eat this tick (food adjacent to its head).
    With ``optimistic=True`` the may-eat case counts as free (used to spot
    cells that are blocked only conservatively).
    """
    board = game_state["board"]
    food = {(f["x"], f["y"]) for f in board["food"]}
    my_id = game_state["you"]["id"]

    blocked: Set[Point] = set()
    for s in board["snakes"]:
        body = snake_body(s)
        blocked.update(body[:-1])
        stacked = len(body) >= 2 and body[-1] == body[-2]
        if stacked:
            blocked.add(body[-1])
            continue
        if s["id"] != my_id and not optimistic:
            ehead = body[0]
            if any((ehead[0] + dx, ehead[1] + dy) in food for dx, dy in DIRECTIONS.values()):
                blocked.add(body[-1])
    return blocked


def legal_moves(game_state: Dict) -> List[str]:
    """Moves that are in bounds and not into an occupied (tail-aware) cell."""
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    blocked = _blocked_cells(game_state)
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in blocked
    ]


def _has_followup(game_state: Dict, move: str) -> bool:
    """After playing ``move``, is there at least one legal move next tick?"""
    sim = simulate_after_move(game_state, move)
    nxt = sim["next_pos"]
    my_body = sim["my_body"]
    vacating: Optional[Point] = None
    if len(my_body) >= 2 and my_body[-1] != my_body[-2]:
        vacating = my_body[-1]
    for dx, dy in DIRECTIONS.values():
        nb = (nxt[0] + dx, nxt[1] + dy)
        if not in_bounds(nb, sim["width"], sim["height"]):
            continue
        if nb not in sim["occupied"] or nb == vacating:
            return True
    return False


def _losing_h2h_cells(game_state: Dict) -> Set[Point]:
    you = game_state["you"]
    return head_to_head_cells(game_state["board"]["snakes"], you["id"], you["length"])


def hard_safety_filter(game_state: Dict, moves: List[str]) -> List[str]:
    """Remove certain-death moves, then losing/tying head-to-head moves.

    Each tier is skipped if it would remove every remaining move.
    """
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])

    survivable = [m for m in moves if _has_followup(game_state, m)]
    if survivable:
        moves = survivable

    danger = _losing_h2h_cells(game_state)
    calm = [
        m
        for m in moves
        if (head[0] + DIRECTIONS[m][0], head[1] + DIRECTIONS[m][1]) not in danger
    ]
    if calm:
        moves = calm
    return moves


def least_bad_move(game_state: Dict) -> str:
    """Best move when nothing is legal: prefer cells blocked only conservatively
    (an enemy tail that may still vacate), then any in-bounds cell over a wall."""
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    blocked = _blocked_cells(game_state)
    blocked_optimistic = _blocked_cells(game_state, optimistic=True)

    best_move, best_score = "up", float("-inf")
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if not in_bounds(nxt, width, height):
            score = 0.0
        elif nxt not in blocked:
            score = 100.0  # legal after all — caller normally handles this case
        elif nxt not in blocked_optimistic:
            score = 50.0  # maybe-vacating tail: only nonzero-chance option
        else:
            score = 1.0  # solid body: certain death, but keep a stable order
        if score > best_score:
            best_move, best_score = move, score
    return best_move
