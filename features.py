"""Shared feature extraction for move ranking.

Single source of truth for both serving (policy.py) and future training
(train.py) — feature parity between train and serve is guaranteed by
importing this module in both places.

All features are computed on the *simulated* next state
(sim.simulate_after_move): tails free correctly, growth keeps the tail.
Food distances use BFS over the simulated map, never Manhattan, so food
behind a wall of bodies is correctly seen as far/unreachable.
"""

from typing import Dict

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
from sim import simulate_after_move

BIG = 10_000
# Below this health we start actively steering toward food.
HUNGRY_THRESHOLD = 50

# The 13 features the embedded linear model was trained on. Order matters for
# nothing here (the scorer looks names up), but the names are frozen.
MODEL_FEATURE_NAMES = [
    "space_capped",
    "open_space",
    "voronoi",
    "reaches_tail",
    "escape",
    "h2h_danger",
    "near_bigger_head",
    "near_enemy_head",
    "wall_dist",
    "food_score",
    "food_delta",
    "is_food",
    "dist_to_center",
]


def candidate_features(state: Dict, move: str) -> Dict[str, float]:
    """Feature vector for playing ``move`` from ``state``. Assumes ``move`` is legal."""
    board = state["board"]
    you = state["you"]
    width, height = board["width"], board["height"]
    my_length = you["length"]
    health = you["health"]
    head = snake_body(you)[0]
    foods = food_points(board)

    sim = simulate_after_move(state, move)
    nxt = sim["next_pos"]
    ate = sim["ate"]
    blocked = sim["occupied"] - {nxt}

    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies]
    bigger_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length]
    danger = head_to_head_cells(board["snakes"], you["id"], my_length)

    # Voronoi control: cells we reach strictly before any enemy.
    my_dist = bfs_dist([nxt], blocked, width, height)
    enemy_dist = bfs_dist(enemy_heads, blocked, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, BIG))

    # Tail reachability is a useful anti-self-trap signal.
    new_tail = sim["my_body"][-1]
    reach = bfs_dist([nxt], blocked - {new_tail}, width, height)
    reaches_tail = 1.0 if new_tail in reach else 0.0

    escape = sum(
        1
        for dx, dy in DIRECTIONS.values()
        if in_bounds((nxt[0] + dx, nxt[1] + dy), width, height)
        and (nxt[0] + dx, nxt[1] + dy) not in blocked
    )

    # Food economics over BFS paths (safe shortest paths, not straight lines).
    bfs_food_next = 0 if ate else min((my_dist.get(f, BIG) for f in foods), default=BIG)
    blocked_now = occupied_cells(board["snakes"]) - {head}
    now_dist = bfs_dist([head], blocked_now, width, height)
    bfs_food_now = min((now_dist.get(f, BIG) for f in foods), default=BIG)

    hungry = health < HUNGRY_THRESHOLD
    reachable = bfs_food_next < BIG
    span = width + height
    if reachable or bfs_food_now < BIG:
        food_delta = float(max(-span, min(span, bfs_food_now - bfs_food_next)))
    else:
        food_delta = 0.0

    return {
        "space_capped": float(flood_fill(nxt, blocked, width, height, limit=my_length + 1)),
        "open_space": float(flood_fill(nxt, blocked, width, height)),
        "voronoi": float(voronoi),
        "reaches_tail": reaches_tail,
        "escape": float(escape),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "near_bigger_head": float(min((manhattan(nxt, h) for h in bigger_heads), default=span)),
        "near_enemy_head": float(min((manhattan(nxt, h) for h in enemy_heads), default=span)),
        "wall_dist": float(min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])),
        "food_score": float((span - bfs_food_next) * 2) if hungry and reachable else 0.0,
        "food_delta": food_delta,
        "is_food": 1.0 if ate else 0.0,
        "dist_to_center": float(abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2)),
        # Extra features (not in the embedded model; used by tests, future
        # training and tie-breaking):
        "bfs_food_dist": float(bfs_food_next),
        "food_reachable": 1.0 if reachable else 0.0,
    }
