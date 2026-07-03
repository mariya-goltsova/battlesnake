"""Minimal standard-rules Battlesnake engine for local games.

Plays policies (plain ``state -> move`` callables) against each other with
no HTTP in between, so the arena can run thousands of seeded, deterministic
games — and later generate self-play training data.

Rule order per turn (mirrors the official rules repo):
  1. gather moves (policy errors / illegal strings count as "up")
  2. move snakes simultaneously (add head, drop tail)
  3. health -= 1
  4. feed (health=100, duplicate tail segment -> stacked tail)
  5. spawn food (seeded RNG)
  6. eliminate: wall / starvation / self-collision / body-collision / head-to-head
"""

import random
import time
from typing import Callable, Dict, List, Optional, Set

from core import DIRECTIONS, Point, in_bounds

# Default start cells for up to four snakes (corner-adjacent, official-like).
_DEFAULT_STARTS: List[Point] = [(1, 1), (-2, -2), (1, -2), (-2, 1)]


def _resolve_start(raw: Point, width: int, height: int) -> Point:
    """Map negative default coordinates onto the far side of the board."""
    x = raw[0] if raw[0] >= 0 else width + raw[0]
    y = raw[1] if raw[1] >= 0 else height + raw[1]
    return (x, y)


def _free_cells(width: int, height: int, snakes: Dict[str, Dict], food: Set[Point]) -> List[Point]:
    taken = set(food)
    for snake in snakes.values():
        taken.update(snake["body"])
    return sorted(
        (x, y) for x in range(width) for y in range(height) if (x, y) not in taken
    )


def _spawn_food(rng: random.Random, food: Set[Point], width: int, height: int,
                snakes: Dict[str, Dict], chance: float, min_food: int) -> None:
    def place_one() -> None:
        free = _free_cells(width, height, snakes, food)
        if free:
            food.add(free[rng.randrange(len(free))])

    if chance > 0 and rng.random() < chance:
        place_one()
    while len(food) < min_food:
        before = len(food)
        place_one()
        if len(food) == before:  # board full
            break


def _api_state(turn: int, width: int, height: int, food: Set[Point],
               snakes: Dict[str, Dict], you_id: str) -> Dict:
    """Build the exact Battlesnake API game-state dict for one snake."""

    def snake_dict(sid: str) -> Dict:
        body = snakes[sid]["body"]
        return {
            "id": sid,
            "health": snakes[sid]["health"],
            "length": len(body),
            "head": {"x": body[0][0], "y": body[0][1]},
            "body": [{"x": x, "y": y} for x, y in body],
        }

    return {
        "turn": turn,
        "board": {
            "width": width,
            "height": height,
            "food": [{"x": x, "y": y} for x, y in sorted(food)],
            "snakes": [snake_dict(sid) for sid in snakes],
        },
        "you": snake_dict(you_id),
    }


def run_game(policies: Dict[str, Callable[[Dict], str]], width: int = 11,
             height: int = 11, seed: int = 0, max_turns: int = 1000,
             starts: Optional[Dict[str, Point]] = None,
             initial_food: Optional[List[Point]] = None,
             food_spawn_chance: float = 0.15, min_food: int = 1,
             start_length: int = 3) -> Dict:
    """Play one seeded game. Returns winner, turns, deaths and latencies.

    ``{"winner": id | None, "turns": int,
       "deaths": {id: {"turn": int, "cause": str}},
       "latency_ms": {id: [float, ...]}}``
    """
    if len(policies) > len(_DEFAULT_STARTS):
        raise ValueError("at most 4 snakes supported")

    rng = random.Random(seed)
    ids = list(policies)

    snakes: Dict[str, Dict] = {}
    for i, sid in enumerate(ids):
        head = (starts or {}).get(sid) or _resolve_start(_DEFAULT_STARTS[i], width, height)
        snakes[sid] = {"health": 100, "body": [head] * start_length}

    food: Set[Point] = set(initial_food) if initial_food is not None else set()
    if initial_food is None:
        _spawn_food(rng, food, width, height, snakes, chance=0.0, min_food=1)

    deaths: Dict[str, Dict] = {}
    latency_ms: Dict[str, List[float]] = {sid: [] for sid in ids}
    end_when = 1 if len(ids) > 1 else 0
    turn = 0

    while turn < max_turns and len(snakes) > end_when:
        turn += 1

        # 1. gather moves from the state *before* this step.
        moves: Dict[str, str] = {}
        for sid in snakes:
            state = _api_state(turn - 1, width, height, food, snakes, sid)
            t0 = time.perf_counter()
            try:
                move = policies[sid](state)
            except Exception:  # noqa: BLE001 - a broken policy forfeits smartly, not the game
                move = "up"
            latency_ms[sid].append((time.perf_counter() - t0) * 1000)
            moves[sid] = move if move in DIRECTIONS else "up"

        # 2. move simultaneously; 3. lose health; 4. feed.
        for sid, snake in snakes.items():
            dx, dy = DIRECTIONS[moves[sid]]
            head = snake["body"][0]
            snake["body"] = [(head[0] + dx, head[1] + dy)] + snake["body"][:-1]
            snake["health"] -= 1
        for sid, snake in snakes.items():
            head = snake["body"][0]
            if head in food:
                food.discard(head)
                snake["health"] = 100
                snake["body"] = snake["body"] + [snake["body"][-1]]

        # 5. spawn food.
        _spawn_food(rng, food, width, height, snakes, food_spawn_chance, min_food)

        # 6. eliminations, judged simultaneously on the post-move board.
        eliminated: Dict[str, str] = {}
        for sid, snake in snakes.items():
            head = snake["body"][0]
            if not in_bounds(head, width, height):
                eliminated[sid] = "wall"
            elif snake["health"] <= 0:
                eliminated[sid] = "starvation"
        for sid, snake in snakes.items():
            if sid in eliminated:
                continue
            head = snake["body"][0]
            if head in snake["body"][1:]:
                eliminated[sid] = "self-collision"
                continue
            for oid, other in snakes.items():
                if oid == sid:
                    continue
                if head == other["body"][0]:
                    if len(snake["body"]) <= len(other["body"]):
                        eliminated[sid] = "head-to-head"
                        break
                elif head in other["body"][1:]:
                    eliminated[sid] = "body-collision"
                    break

        for sid, cause in eliminated.items():
            deaths[sid] = {"turn": turn, "cause": cause}
            del snakes[sid]

    if len(snakes) == 1:
        winner: Optional[str] = next(iter(snakes))
    elif len(snakes) > 1:  # max_turns reached: longest snake wins, ties draw
        best = max(len(s["body"]) for s in snakes.values())
        leaders = [sid for sid, s in snakes.items() if len(s["body"]) == best]
        winner = leaders[0] if len(leaders) == 1 else None
    else:
        winner = None

    return {"winner": winner, "turns": turn, "deaths": deaths, "latency_ms": latency_ms}
