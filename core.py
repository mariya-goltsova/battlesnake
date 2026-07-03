"""Shared board primitives for the src_rl package.

Semantics match the frozen legacy implementation (baselines/legacy_policy.py)
exactly — verified by parity tests. Board coordinates: ``(0, 0)`` is the
bottom-left corner; up -> y+1, down -> y-1, left -> x-1, right -> x+1.
"""

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


def in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_dist(
    sources: Iterable[Point], blocked: Set[Point], width: int, height: int
) -> Dict[Point, int]:
    """Shortest free-cell distances from seed cells.

    Sources are seeded at distance 0 even if listed in ``blocked``; expansion
    only enters cells not in ``blocked``. Unreachable cells are absent.
    """
    dist: Dict[Point, int] = {}
    dq: deque = deque()
    for source in sources:
        if source not in dist:
            dist[source] = 0
            dq.append(source)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in DIRECTIONS.values():
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in blocked and nb not in dist:
                dist[nb] = d + 1
                dq.append(nb)
    return dist


def flood_fill(
    start: Point,
    occupied: Set[Point],
    width: int,
    height: int,
    limit: Optional[int] = None,
) -> int:
    """Count open cells reachable from ``start``, capped at ``limit`` if given.

    ``start`` itself is counted and assumed enterable (callers pass the cell
    they are about to move into).
    """
    cap = limit if limit is not None else width * height
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= cap:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not in_bounds(nbr, width, height):
                continue
            if nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


def snake_body(snake: Dict) -> List[Point]:
    return [(seg["x"], seg["y"]) for seg in snake["body"]]


def food_points(board: Dict) -> List[Point]:
    return [(f["x"], f["y"]) for f in board["food"]]


def occupied_cells(snakes: List[Dict]) -> Set[Point]:
    """All cells currently filled by any snake's body, tails included."""
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Cells adjacent to heads of enemies with ``length >= my_length``."""
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] < my_length:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger
