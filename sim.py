"""Battlesnake standard-rules game engine.

Mirrors the official rules pipeline (github.com/BattlesnakeOfficial/rules,
standard ruleset) exactly, in this per-tick order:

  1. move snakes  2. reduce health  3. feed  4. spawn food  5. eliminate

Elimination is two-phase like the official engine: phase A (out-of-health,
wall) applies first; phase B collisions (self, body, head-to-head) are computed
simultaneously among phase-A survivors only, then applied together. Lengths for
head-to-head resolution are post-feed.

Internal state is a plain picklable dict; ``make_game_state`` renders the exact
Battlesnake API view so the same policy functions run live and in simulation.
``step`` is functional (returns a new state); the RNG object is shared with the
input state and advances — recreate a game from its seed for reproducibility.
"""

import random
from typing import Dict, List, Optional, Set, Tuple

from src_rl.core import DIRECTIONS, Point, in_bounds

DEFAULT_SETTINGS = {"food_spawn_chance": 15, "minimum_food": 1}


# --- state construction ---------------------------------------------------------


def _start_points(width: int, height: int) -> List[Point]:
    """Official fixed start positions: corners + cardinal midpoints (odd coords)."""
    corners = [(1, 1), (1, height - 2), (width - 2, 1), (width - 2, height - 2)]
    cardinals = [(1, height // 2), (width // 2, 1), (width - 2, height // 2), (width // 2, height - 2)]
    return corners + cardinals


def init_game(
    width: int = 11,
    height: int = 11,
    n_snakes: int = 4,
    seed: Optional[int] = None,
    snake_ids: Optional[List[str]] = None,
) -> Dict:
    """Official-style start: length-3 snakes stacked on fixed points, one food
    diagonally adjacent to each snake, plus one food at the center."""
    rng = random.Random(seed)
    ids = snake_ids or [f"snake-{i}" for i in range(n_snakes)]
    if len(ids) != n_snakes:
        raise ValueError("snake_ids length must match n_snakes")

    starts = rng.sample(_start_points(width, height), n_snakes)
    snakes = [
        {
            "id": sid,
            "health": 100,
            "body": [pos, pos, pos],
            "alive": True,
            "death_cause": None,
            "death_turn": None,
        }
        for sid, pos in zip(ids, starts)
    ]

    center = (width // 2, height // 2)
    food: Set[Point] = {center}
    taken = set(starts) | food
    for pos in starts:
        options = [
            (pos[0] + dx, pos[1] + dy)
            for dx in (-1, 1)
            for dy in (-1, 1)
            if in_bounds((pos[0] + dx, pos[1] + dy), width, height)
            and (pos[0] + dx, pos[1] + dy) not in taken
        ]
        if options:
            cell = rng.choice(sorted(options))
            food.add(cell)
            taken.add(cell)

    return {
        "width": width,
        "height": height,
        "turn": 0,
        "food": food,
        "snakes": snakes,
        "settings": dict(DEFAULT_SETTINGS),
        "rng": rng,
    }


# --- the tick pipeline ------------------------------------------------------------


def _default_direction(body: List[Point]) -> str:
    """Direction the snake is already traveling (used when a move is missing)."""
    if len(body) >= 2:
        dx, dy = body[0][0] - body[1][0], body[0][1] - body[1][1]
        for move, delta in DIRECTIONS.items():
            if delta == (dx, dy):
                return move
    return "up"


def step(state: Dict, moves: Dict[str, str]) -> Dict:
    """Advance one tick. ``moves`` maps snake id -> direction; a missing or
    invalid move repeats the snake's current direction (official engine
    behavior for timeouts)."""
    width, height = state["width"], state["height"]
    settings = state["settings"]
    rng = state["rng"]
    turn = state["turn"] + 1

    snakes = [dict(s, body=list(s["body"])) for s in state["snakes"]]
    food = set(state["food"])

    # 1. Move
    for s in snakes:
        if not s["alive"]:
            continue
        move = moves.get(s["id"])
        if move not in DIRECTIONS:
            move = _default_direction(s["body"])
        dx, dy = DIRECTIONS[move]
        head = s["body"][0]
        s["body"] = [(head[0] + dx, head[1] + dy)] + s["body"][:-1]

    # 2. Reduce health
    for s in snakes:
        if s["alive"]:
            s["health"] -= 1

    # 3. Feed (all heads on a food cell eat; lengths grow before elimination)
    eaten: Set[Point] = set()
    for s in snakes:
        if s["alive"] and s["body"][0] in food:
            s["health"] = 100
            s["body"].append(s["body"][-1])
            eaten.add(s["body"][0])
    food -= eaten

    # 4. Spawn food
    if settings["minimum_food"] > 0 or settings["food_spawn_chance"] > 0:
        occupied = {c for s in snakes if s["alive"] for c in s["body"]}
        free = sorted(
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) not in occupied and (x, y) not in food
        )
        need = 0
        if len(food) < settings["minimum_food"]:
            need = settings["minimum_food"] - len(food)
        elif settings["food_spawn_chance"] > 0 and rng.random() < settings["food_spawn_chance"] / 100.0:
            need = 1
        for _ in range(min(need, len(free))):
            cell = free[rng.randrange(len(free))]
            free.remove(cell)
            food.add(cell)

    # 5a. Eliminate: out of health, out of bounds
    for s in snakes:
        if not s["alive"]:
            continue
        if s["health"] <= 0:
            _kill(s, "out-of-health", turn)
        elif not in_bounds(s["body"][0], width, height):
            _kill(s, "wall", turn)

    # 5b. Eliminate: collisions, computed simultaneously among 5a survivors
    survivors = [s for s in snakes if s["alive"]]
    deaths: List[Tuple[Dict, str]] = []
    for s in survivors:
        head = s["body"][0]
        cause = None
        if head in s["body"][1:]:
            cause = "self-collision"
        if cause is None:
            for o in survivors:
                if o is not s and head in o["body"][1:]:
                    cause = "body-collision"
                    break
        if cause is None:
            for o in survivors:
                if o is not s and head == o["body"][0] and len(o["body"]) >= len(s["body"]):
                    cause = "head-to-head"
                    break
        if cause is not None:
            deaths.append((s, cause))
    for s, cause in deaths:
        _kill(s, cause, turn)

    return {
        "width": width,
        "height": height,
        "turn": turn,
        "food": food,
        "snakes": snakes,
        "settings": settings,
        "rng": rng,
    }


def _kill(snake: Dict, cause: str, turn: int) -> None:
    snake["alive"] = False
    snake["death_cause"] = cause
    snake["death_turn"] = turn


# --- game status -------------------------------------------------------------------


def alive_snakes(state: Dict) -> List[Dict]:
    return [s for s in state["snakes"] if s["alive"]]


def game_over(state: Dict) -> bool:
    alive = len(alive_snakes(state))
    return alive == 0 if len(state["snakes"]) == 1 else alive <= 1


def winner(state: Dict) -> Optional[str]:
    """Id of the sole survivor of a finished multi-snake game, else None."""
    if len(state["snakes"]) == 1:
        return None
    alive = alive_snakes(state)
    return alive[0]["id"] if len(alive) == 1 else None


# --- API view ----------------------------------------------------------------------


def _api_snake(s: Dict) -> Dict:
    body = [{"x": x, "y": y} for x, y in s["body"]]
    return {
        "id": s["id"],
        "name": s["id"],
        "health": s["health"],
        "body": body,
        "head": body[0],
        "length": len(body),
        "latency": "0",
        "shout": "",
    }


def make_game_state(state: Dict, snake_id: str) -> Dict:
    """Render the exact Battlesnake API game_state for one snake's point of view."""
    you = next(s for s in state["snakes"] if s["id"] == snake_id)
    return {
        "game": {
            "id": "sim-game",
            "ruleset": {
                "name": "standard",
                "version": "src_rl-sim",
                "settings": {
                    "foodSpawnChance": state["settings"]["food_spawn_chance"],
                    "minimumFood": state["settings"]["minimum_food"],
                    "hazardDamagePerTurn": 0,
                },
            },
            "map": "standard",
            "timeout": 500,
            "source": "sim",
        },
        "turn": state["turn"],
        "board": {
            "width": state["width"],
            "height": state["height"],
            "food": [{"x": x, "y": y} for x, y in sorted(state["food"])],
            "hazards": [],
            "snakes": [_api_snake(s) for s in state["snakes"] if s["alive"]],
        },
        "you": _api_snake(you),
    }


# --- 1-ply lookahead for feature computation (serve-side) ----------------------------


def simulate_after_move(game_state: Dict, move: str) -> Dict:
    """Occupancy after we play ``move``, with correct tail handling.

    Our tail vacates unless we land on food. Enemy tails vacate unless food is
    adjacent to their head (they might eat and keep it — conservative). A tail
    stacked from eating last turn stays occupied either way, since dropping one
    duplicate still leaves the segment in place.
    """
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    food = {(f["x"], f["y"]) for f in board["food"]}

    head = (you["head"]["x"], you["head"]["y"])
    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)
    ate = nxt in food

    my_body = [(seg["x"], seg["y"]) for seg in you["body"]]
    my_body = [nxt] + (my_body if ate else my_body[:-1])

    occupied: Set[Point] = set(my_body)
    for s in board["snakes"]:
        if s["id"] == you["id"]:
            continue
        body = [(seg["x"], seg["y"]) for seg in s["body"]]
        ehead = body[0]
        may_eat = any(
            (ehead[0] + ddx, ehead[1] + ddy) in food for ddx, ddy in DIRECTIONS.values()
        )
        occupied.update(body if may_eat else body[:-1])

    return {
        "next_pos": nxt,
        "ate": ate,
        "my_body": my_body,
        "occupied": occupied,
        "width": width,
        "height": height,
    }
