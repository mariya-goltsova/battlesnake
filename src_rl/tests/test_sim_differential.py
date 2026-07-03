"""Differential tests: replay official-CLI games through sim.step and compare.

For every consecutive turn pair in a recorded game we rebuild the internal
state from the CLI's turn N line, infer each surviving snake's move from its
head displacement, and check that our step() reproduces turn N+1 exactly
(bodies, healths, eliminations). Snakes eliminated at N+1 vanish from the CLI
output, so their move is unknown — we accept the transition if ANY combination
of their possible moves reproduces the observed board.

Food spawn locations depend on the official engine's RNG, which we cannot
reproduce. We therefore replay with spawning disabled and require:
our post-feed food == CLI food minus at most `max_spawn` new cells.
With foodSpawnChance=0 and minimumFood=0 fixtures this becomes an exact match.
"""

import itertools
import json
import random
from pathlib import Path

import pytest

from src_rl import sim
from src_rl.core import DIRECTIONS

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "cli_games").glob("*.jsonl"))


def load_game(path):
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    meta, turns, result = lines[0], lines[1:-1], lines[-1]
    assert "winnerName" in result or result.get("isDraw") is not None
    return meta, turns


def internal_state(turn_line, settings):
    board = turn_line["board"]
    return {
        "width": board["width"],
        "height": board["height"],
        "turn": turn_line["turn"],
        "food": {(f["x"], f["y"]) for f in board.get("food", [])},
        "snakes": [
            {
                "id": s["name"],  # names are stable and unique in our fixtures
                "health": s["health"],
                "body": [(seg["x"], seg["y"]) for seg in s["body"]],
                "alive": True,
                "death_cause": None,
                "death_turn": None,
            }
            for s in board["snakes"]
        ],
        "settings": settings,
        "rng": random.Random(0),
    }


def infer_move(head_before, head_after):
    delta = (head_after[0] - head_before[0], head_after[1] - head_before[1])
    for move, d in DIRECTIONS.items():
        if d == delta:
            return move
    raise AssertionError(f"non-adjacent head displacement {head_before} -> {head_after}")


def snakes_by_name(turn_line):
    return {
        s["name"]: {
            "health": s["health"],
            "body": [(seg["x"], seg["y"]) for seg in s["body"]],
        }
        for s in turn_line["board"]["snakes"]
    }


def check_transition(prev_line, next_line, settings):
    no_spawn = {"food_spawn_chance": 0, "minimum_food": 0}
    prev, nxt = snakes_by_name(prev_line), snakes_by_name(next_line)
    survivors = {n: infer_move(prev[n]["body"][0], nxt[n]["body"][0]) for n in prev if n in nxt}
    eliminated = [n for n in prev if n not in nxt]
    next_food = {(f["x"], f["y"]) for f in next_line["board"].get("food", [])}
    max_spawn = max(settings["minimum_food"], 1 if settings["food_spawn_chance"] > 0 else 0)

    failures = []
    for extra in itertools.product(*[[(n, m) for m in DIRECTIONS] for n in eliminated]):
        moves = dict(survivors, **dict(extra))
        out = sim.step(internal_state(prev_line, no_spawn), moves)
        ours = {s["id"]: s for s in out["snakes"]}

        mismatch = None
        for name, exp in nxt.items():
            s = ours[name]
            if not s["alive"]:
                mismatch = f"{name} died in sim ({s['death_cause']}) but survived in CLI"
            elif s["body"] != exp["body"]:
                mismatch = f"{name} body {s['body']} != {exp['body']}"
            elif s["health"] != exp["health"]:
                mismatch = f"{name} health {s['health']} != {exp['health']}"
            if mismatch:
                break
        if mismatch is None:
            for name in eliminated:
                if ours[name]["alive"]:
                    mismatch = f"{name} survived in sim but was eliminated in CLI"
                    break
        if mismatch is None:
            missing = out["food"] - next_food
            surplus = next_food - out["food"]
            if missing:
                mismatch = f"sim kept food the CLI removed: {sorted(missing)}"
            elif len(surplus) > max_spawn:
                mismatch = f"CLI has {len(surplus)} new food, engine could spawn <= {max_spawn}"
        if mismatch is None:
            return
        failures.append(mismatch)

    raise AssertionError(
        f"turn {prev_line['turn']} -> {next_line['turn']} "
        f"(eliminated: {eliminated or 'none'}): {failures[:3]}"
    )


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_replay_official_game(path):
    meta, turns = load_game(path)
    s = meta["ruleset"]["settings"]
    settings = {"food_spawn_chance": s["foodSpawnChance"], "minimum_food": s["minimumFood"]}
    assert meta["ruleset"]["name"] == "standard"
    assert len(turns) >= 2, "fixture too short"
    for prev_line, next_line in zip(turns, turns[1:]):
        check_transition(prev_line, next_line, settings)


def test_fixtures_exist():
    assert len(FIXTURES) >= 12, (
        "CLI game fixtures missing — run: python -m src_rl.tests.record_cli_games"
    )
