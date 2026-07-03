"""The single feature function shared by serving and training (parity by construction).

Features are computed on the *simulated* post-move board (sim.simulate_after_move):
tails vacate correctly, our body has already advanced, and food distances use
BFS on the real occupancy — the three P0 fixes over the legacy feature set.
Legacy's 13 feature names are kept (semantics improved) plus three new ones.
"""

from typing import Dict, List, Optional, Set, Tuple

from src_rl.core import DIRECTIONS, Point, bfs_dist, flood_fill, in_bounds, manhattan
from src_rl.sim import simulate_after_move

# Sentinel BFS distance for unreachable food; bounded so standardization stays sane.
UNREACHABLE = 128.0

HUNGRY_THRESHOLD = 50

FEATURE_NAMES: List[str] = [
    # legacy names, corrected semantics
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
    # new
    "bfs_food_dist",
    "food_reachable",
    "length_advantage",
]


def build_context(game_state: Dict) -> Dict:
    """Per-state precomputation shared by all candidate moves."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    foods: Set[Point] = {(f["x"], f["y"]) for f in board["food"]}
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]

    current_blocked = {
        (seg["x"], seg["y"]) for s in board["snakes"] for seg in s["body"]
    }
    if foods:
        now_dist = bfs_dist([head], current_blocked, width, height)
        nearest_now = min((now_dist[f] for f in foods if f in now_dist), default=UNREACHABLE)
    else:
        nearest_now = UNREACHABLE

    return {
        "foods": foods,
        "enemies": enemies,
        "enemy_heads": [(s["head"]["x"], s["head"]["y"]) for s in enemies],
        "enemy_lengths": [s["length"] for s in enemies],
        "nearest_food_now": nearest_now,
    }


def candidate_features(game_state: Dict, move: str, ctx: Optional[Dict] = None) -> Dict[str, float]:
    """Feature vector for playing ``move`` from ``game_state``. Assumes ``move``
    is in bounds; pass a shared ``ctx`` from build_context to amortize per-state work."""
    if ctx is None:
        ctx = build_context(game_state)
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    health = you["health"]

    sim = simulate_after_move(game_state, move)
    nxt: Point = sim["next_pos"]
    my_body: List[Point] = sim["my_body"]
    occupied: Set[Point] = sim["occupied"]
    my_length = len(my_body)

    # Reachability from the simulated position (standing cell excluded from counts,
    # matching legacy's magnitudes where the entered cell was still free).
    open_space = flood_fill(nxt, occupied, width, height) - 1
    cap = my_length + 1
    space_capped = flood_fill(nxt, occupied, width, height, limit=cap + 1) - 1

    my_dist = bfs_dist([nxt], occupied, width, height)
    enemy_heads = ctx["enemy_heads"]
    if enemy_heads:
        enemy_dist = bfs_dist(enemy_heads, occupied, width, height)
        voronoi = sum(
            1 for cell, md in my_dist.items() if cell != nxt and md < enemy_dist.get(cell, UNREACHABLE)
        )
    else:
        voronoi = len(my_dist) - 1

    my_tail = my_body[-1]
    reach = bfs_dist([nxt], occupied - {my_tail}, width, height)
    reaches_tail = 1.0 if my_tail in reach else 0.0

    escape = sum(
        1
        for dx, dy in DIRECTIONS.values()
        if in_bounds((nxt[0] + dx, nxt[1] + dy), width, height)
        and (nxt[0] + dx, nxt[1] + dy) not in occupied
    )

    # Head-to-head with post-feed length: enemies whose length matches ours after
    # we (possibly) eat are still deadly ties.
    bigger_heads = [
        h for h, ln in zip(enemy_heads, ctx["enemy_lengths"]) if ln >= my_length
    ]
    danger = {
        (h[0] + dx, h[1] + dy) for h in bigger_heads for dx, dy in DIRECTIONS.values()
    }

    foods = ctx["foods"]
    if foods:
        nearest_next = min((my_dist[f] for f in foods if f in my_dist), default=UNREACHABLE)
    else:
        nearest_next = UNREACHABLE
    food_reachable = 1.0 if nearest_next < UNREACHABLE else 0.0
    hungry = health < HUNGRY_THRESHOLD

    max_enemy_len = max(ctx["enemy_lengths"], default=0)

    return {
        "space_capped": float(space_capped),
        "open_space": float(open_space),
        "voronoi": float(voronoi),
        "reaches_tail": reaches_tail,
        "escape": float(escape),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "near_bigger_head": float(
            min((manhattan(nxt, h) for h in bigger_heads), default=width + height)
        ),
        "near_enemy_head": float(
            min((manhattan(nxt, h) for h in enemy_heads), default=width + height)
        ),
        "wall_dist": float(min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])),
        "food_score": float((width + height - nearest_next) * 2)
        if hungry and food_reachable
        else 0.0,
        "food_delta": float(ctx["nearest_food_now"] - nearest_next) if foods else 0.0,
        "is_food": 1.0 if sim["ate"] else 0.0,
        "dist_to_center": abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2),
        "bfs_food_dist": float(nearest_next),
        "food_reachable": food_reachable,
        "length_advantage": float(my_length - max_enemy_len),
    }
