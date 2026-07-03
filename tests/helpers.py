"""Shared fixtures builders for the test suite."""

from typing import Dict, List, Tuple

Point = Tuple[int, int]


def make_snake(snake_id: str, body: List[Point], health: int = 100) -> Dict:
    """Build an API-schema snake dict from a list of (x, y) points, head first."""
    return {
        "id": snake_id,
        "health": health,
        "length": len(body),
        "head": {"x": body[0][0], "y": body[0][1]},
        "body": [{"x": x, "y": y} for x, y in body],
    }


def make_state(you: Dict, snakes: List[Dict] = None, food: List[Point] = (),
               width: int = 11, height: int = 11, turn: int = 1) -> Dict:
    """Build an API-schema game state. `you` must also appear in `snakes`
    (it is added automatically when `snakes` is None)."""
    if snakes is None:
        snakes = [you]
    return {
        "turn": turn,
        "board": {
            "width": width,
            "height": height,
            "food": [{"x": x, "y": y} for x, y in food],
            "snakes": snakes,
        },
        "you": you,
    }
