"""Builders for API-schema game states used across the test suite."""

from typing import Dict, List, Optional, Tuple

Point = Tuple[int, int]


def make_snake(snake_id: str, body: List[Point], health: int = 100) -> Dict:
    """Build an API-schema snake dict from a list of (x, y) points, head first."""
    return {
        "id": snake_id,
        "name": snake_id,
        "health": health,
        "length": len(body),
        "head": {"x": body[0][0], "y": body[0][1]},
        "body": [{"x": x, "y": y} for x, y in body],
        "latency": "0",
        "shout": "",
    }


def make_state(
    you: Dict,
    snakes: Optional[List[Dict]] = None,
    food: Optional[List[Point]] = None,
    width: int = 11,
    height: int = 11,
    turn: int = 1,
) -> Dict:
    """Build an API-schema game_state. ``you`` must also appear in ``snakes``
    (it is added automatically when ``snakes`` is omitted)."""
    all_snakes = snakes if snakes is not None else [you]
    return {
        "game": {
            "id": "test-game",
            "ruleset": {
                "name": "standard",
                "version": "test",
                "settings": {"foodSpawnChance": 15, "minimumFood": 1, "hazardDamagePerTurn": 0},
            },
            "map": "standard",
            "timeout": 500,
            "source": "test",
        },
        "turn": turn,
        "board": {
            "width": width,
            "height": height,
            "food": [{"x": x, "y": y} for x, y in (food or [])],
            "hazards": [],
            "snakes": all_snakes,
        },
        "you": you,
    }
