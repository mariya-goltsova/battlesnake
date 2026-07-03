"""Tests for engine.py — the minimal standard-rules Battlesnake engine."""

from engine import run_game


def scripted(moves):
    """Policy that replays a fixed move sequence (repeats the last move)."""
    it = iter(moves)
    last = moves[-1]

    def policy(_state):
        nonlocal last
        try:
            last = next(it)
        except StopIteration:
            pass
        return last

    return policy


def recording(moves, log):
    """Scripted policy that also records every state it is called with."""
    inner = scripted(moves)

    def policy(state):
        log.append(state)
        return inner(state)

    return policy


def test_wall_death_at_expected_turn():
    result = run_game(
        {"a": scripted(["up"])},
        width=5, height=5, seed=0,
        starts={"a": (1, 1)}, initial_food=[], food_spawn_chance=0.0, min_food=0,
    )
    # (1,1) -> (1,2) -> (1,3) -> (1,4) -> off the board on step 4.
    assert result["deaths"]["a"] == {"turn": 4, "cause": "wall"}
    assert result["winner"] is None
    assert result["turns"] == 4
    assert len(result["latency_ms"]["a"]) == 4


def test_eating_grows_heals_and_stacks_tail_for_one_turn():
    log = []
    run_game(
        {"a": recording(["up"], log)},
        width=7, height=7, seed=0,
        starts={"a": (2, 2)}, initial_food=[(2, 5)], food_spawn_chance=0.0, min_food=0,
    )
    # Steps: 1 -> (2,3), 2 -> (2,4), 3 -> (2,5) EAT, 4 -> (2,6): observe stacked tail.
    s3 = log[3]["you"]  # state before step 4, i.e. right after eating
    body3 = [(seg["x"], seg["y"]) for seg in s3["body"]]
    assert s3["health"] == 100
    assert s3["length"] == 4
    assert body3 == [(2, 5), (2, 4), (2, 3), (2, 3)]  # duplicated tail
    assert log[3]["board"]["food"] == []  # food consumed

    s4 = log[4]["you"]  # one turn later the tail has unstacked
    body4 = [(seg["x"], seg["y"]) for seg in s4["body"]]
    assert body4 == [(2, 6), (2, 5), (2, 4), (2, 3)]
    assert s4["health"] == 99


def test_starvation_at_health_zero():
    # A 2x2 loop never eats; health hits 0 after 100 steps.
    loop = ["right", "up", "left", "down"] * 30
    result = run_game(
        {"a": scripted(loop)},
        width=7, height=7, seed=0,
        starts={"a": (2, 2)}, initial_food=[], food_spawn_chance=0.0, min_food=0,
        max_turns=200,
    )
    assert result["deaths"]["a"] == {"turn": 100, "cause": "starvation"}


def test_self_collision_cause():
    # Grow straight up with start_length=5, then reverse into our own neck.
    result = run_game(
        {"a": scripted(["up", "up", "up", "up", "down"])},
        width=7, height=7, seed=0, start_length=5,
        starts={"a": (2, 2)}, initial_food=[], food_spawn_chance=0.0, min_food=0,
    )
    assert result["deaths"]["a"]["cause"] == "self-collision"
    assert result["deaths"]["a"]["turn"] == 5


def test_body_collision_cause():
    # "a" walks right into "b"'s still-occupied cell.
    result = run_game(
        {"a": scripted(["right"]), "b": scripted(["right", "up", "left", "down"])},
        width=7, height=7, seed=0,
        starts={"a": (1, 3), "b": (3, 3)}, initial_food=[], food_spawn_chance=0.0, min_food=0,
    )
    assert result["deaths"]["a"]["cause"] == "body-collision"
    assert result["winner"] == "b"


def test_head_to_head_equal_lengths_both_die():
    result = run_game(
        {"a": scripted(["right"]), "b": scripted(["left"])},
        width=7, height=7, seed=0,
        starts={"a": (1, 3), "b": (5, 3)}, initial_food=[], food_spawn_chance=0.0, min_food=0,
    )
    assert result["deaths"]["a"]["cause"] == "head-to-head"
    assert result["deaths"]["b"]["cause"] == "head-to-head"
    assert result["deaths"]["a"]["turn"] == 2
    assert result["winner"] is None


def test_head_to_head_longer_snake_survives():
    # "a" eats on step 1 (length 4) before the step-2 head-to-head.
    result = run_game(
        {"a": scripted(["right"]), "b": scripted(["left"])},
        width=7, height=7, seed=0,
        starts={"a": (1, 3), "b": (5, 3)}, initial_food=[(2, 3)],
        food_spawn_chance=0.0, min_food=0,
    )
    assert result["deaths"]["b"]["cause"] == "head-to-head"
    assert "a" not in result["deaths"]
    assert result["winner"] == "a"


def test_same_seed_same_result():
    def game():
        return run_game(
            {"a": scripted(["right", "up", "left", "down"] * 40),
             "b": scripted(["left", "down", "right", "up"] * 40)},
            width=11, height=11, seed=42, max_turns=150,
        )

    r1, r2 = game(), game()
    r1.pop("latency_ms"), r2.pop("latency_ms")  # wall-clock noise
    assert r1 == r2
