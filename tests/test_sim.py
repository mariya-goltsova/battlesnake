"""Rule-by-rule unit tests for the sim.py game engine (standard Battlesnake rules)."""

import random

from src_rl import sim


def snake(sid, body, health=100):
    """Internal-state snake (body is a list of (x, y), head first)."""
    return {
        "id": sid,
        "health": health,
        "body": list(body),
        "alive": True,
        "death_cause": None,
        "death_turn": None,
    }


def state(snakes, food=(), width=11, height=11, turn=0, chance=15, minimum=1, seed=0):
    return {
        "width": width,
        "height": height,
        "turn": turn,
        "food": set(food),
        "snakes": snakes,
        "settings": {"food_spawn_chance": chance, "minimum_food": minimum},
        "rng": random.Random(seed),
    }


def get(st, sid):
    return next(s for s in st["snakes"] if s["id"] == sid)


NO_FOOD = dict(chance=0, minimum=0)


# --- movement, health, feeding ------------------------------------------------------


def test_move_advances_head_and_vacates_tail():
    st = state([snake("a", [(5, 5), (5, 6), (5, 7)])], **NO_FOOD)
    nxt = sim.step(st, {"a": "right"})
    a = get(nxt, "a")
    assert a["body"] == [(6, 5), (5, 5), (5, 6)]
    assert a["health"] == 99
    assert nxt["turn"] == 1
    # original state untouched (functional step)
    assert get(st, "a")["body"] == [(5, 5), (5, 6), (5, 7)]


def test_eating_resets_health_grows_and_stacks_tail():
    st = state([snake("a", [(5, 5), (5, 6), (5, 7)], health=42)], food=[(6, 5)], **NO_FOOD)
    nxt = sim.step(st, {"a": "right"})
    a = get(nxt, "a")
    assert a["health"] == 100
    assert len(a["body"]) == 4
    assert a["body"][-1] == a["body"][-2]  # duplicated tail segment
    assert (6, 5) not in nxt["food"]
    # the stacked tail unfolds on the following move
    nxt2 = sim.step(nxt, {"a": "right"})
    a2 = get(nxt2, "a")
    assert len(a2["body"]) == 4
    assert a2["body"][-1] != a2["body"][-2]


def test_starvation_eliminates_at_zero_health():
    st = state([snake("a", [(5, 5), (5, 6), (5, 7)], health=1)], **NO_FOOD)
    nxt = sim.step(st, {"a": "right"})
    a = get(nxt, "a")
    assert not a["alive"]
    assert a["death_cause"] == "out-of-health"
    assert a["death_turn"] == 1


def test_eating_on_last_health_point_survives():
    st = state([snake("a", [(5, 5), (5, 6), (5, 7)], health=1)], food=[(6, 5)], **NO_FOOD)
    a = get(sim.step(st, {"a": "right"}), "a")
    assert a["alive"] and a["health"] == 100


# --- collisions ---------------------------------------------------------------------


def test_wall_collision():
    st = state([snake("a", [(10, 5), (9, 5), (8, 5)])], **NO_FOOD)
    a = get(sim.step(st, {"a": "right"}), "a")
    assert not a["alive"] and a["death_cause"] == "wall"


def test_self_collision_moving_into_own_neck():
    st = state([snake("a", [(5, 5), (5, 6), (5, 7), (6, 7)])], **NO_FOOD)
    a = get(sim.step(st, {"a": "up"}), "a")
    assert not a["alive"] and a["death_cause"] == "self-collision"


def test_body_collision_with_enemy():
    st = state(
        [snake("a", [(4, 5), (3, 5), (2, 5)]), snake("b", [(5, 6), (5, 7), (5, 8), (5, 9)])],
        **NO_FOOD,
    )
    # b moves down to (5,5); a moves right to (5,5)?? -> that's h2h. Instead:
    # a moves right into b's body cell (5,6)->(5,5) after b moves? b's body after
    # moving down occupies (5,5),(5,6),(5,7),(5,8). a moving right lands on (5,5)
    # which is b's HEAD -> h2h. Use a target further up b's body instead:
    st = state(
        [snake("a", [(4, 6), (3, 6), (2, 6)]), snake("b", [(5, 5), (5, 6), (5, 7), (5, 8)])],
        **NO_FOOD,
    )
    nxt = sim.step(st, {"a": "right", "b": "down"})  # a -> (5,6): b's body
    a, b = get(nxt, "a"), get(nxt, "b")
    assert not a["alive"] and a["death_cause"] == "body-collision"
    assert b["alive"]


def test_tail_chase_is_legal():
    # a follows directly behind b; b's tail vacates the same tick a enters it.
    st = state(
        [snake("a", [(4, 5), (3, 5), (2, 5)]), snake("b", [(7, 5), (6, 5), (5, 5)])],
        **NO_FOOD,
    )
    nxt = sim.step(st, {"a": "right", "b": "right"})  # a -> (5,5) just vacated
    assert get(nxt, "a")["alive"] and get(nxt, "b")["alive"]


def test_tail_chase_after_enemy_ate_kills():
    # b ate last tick -> stacked tail at (5,5): the cell does NOT vacate this tick.
    st = state(
        [snake("a", [(4, 5), (3, 5), (2, 5)]), snake("b", [(7, 5), (6, 5), (5, 5), (5, 5)])],
        **NO_FOOD,
    )
    nxt = sim.step(st, {"a": "right", "b": "right"})
    a = get(nxt, "a")
    assert not a["alive"] and a["death_cause"] == "body-collision"


def test_head_to_head_shorter_dies():
    st = state(
        [snake("a", [(4, 5), (3, 5)]), snake("b", [(6, 5), (7, 5), (8, 5)])],
        **NO_FOOD,
    )
    nxt = sim.step(st, {"a": "right", "b": "left"})  # both -> (5,5)
    a, b = get(nxt, "a"), get(nxt, "b")
    assert not a["alive"] and a["death_cause"] == "head-to-head"
    assert b["alive"]


def test_head_to_head_equal_lengths_both_die():
    st = state(
        [snake("a", [(4, 5), (3, 5), (2, 5)]), snake("b", [(6, 5), (7, 5), (8, 5)])],
        **NO_FOOD,
    )
    nxt = sim.step(st, {"a": "right", "b": "left"})
    assert not get(nxt, "a")["alive"] and not get(nxt, "b")["alive"]


def test_double_eat_then_head_to_head_uses_post_feed_lengths():
    # Both heads land on the same food cell: both eat first (feed phase), THEN
    # the head-to-head resolves with post-feed lengths (a: 2->3, b: 3->4).
    st = state(
        [snake("a", [(4, 5), (3, 5)]), snake("b", [(6, 5), (7, 5), (8, 5)])],
        food=[(5, 5)],
        **NO_FOOD,
    )
    nxt = sim.step(st, {"a": "right", "b": "left"})
    a, b = get(nxt, "a"), get(nxt, "b")
    assert not a["alive"] and a["death_cause"] == "head-to-head"
    assert b["alive"] and b["health"] == 100 and len(b["body"]) == 4


def test_starved_snake_is_not_a_collision_target_same_tick():
    # b starves this tick (phase A); a moving into b's body must survive (phase B
    # only considers phase-A survivors) — mirrors the official engine.
    st = state(
        [snake("a", [(4, 6), (3, 6), (2, 6)]), snake("b", [(5, 5), (5, 6), (5, 7), (5, 8)], health=1)],
        **NO_FOOD,
    )
    nxt = sim.step(st, {"a": "right", "b": "down"})
    a, b = get(nxt, "a"), get(nxt, "b")
    assert not b["alive"] and b["death_cause"] == "out-of-health"
    assert a["alive"]


# --- food spawning ------------------------------------------------------------------


def test_minimum_food_respawns_deterministically():
    st = state([snake("a", [(5, 5), (5, 6), (5, 7)])], food=[(6, 5)], chance=0, minimum=1)
    nxt = sim.step(st, {"a": "right"})  # eats the only food
    assert len(nxt["food"]) == 1  # respawned to satisfy minimum_food
    assert next(iter(nxt["food"])) not in {(6, 5), (5, 5), (5, 6)}


def test_no_spawn_when_chance_zero_and_minimum_satisfied():
    st = state([snake("a", [(5, 5), (5, 6), (5, 7)])], food=[(0, 0)], chance=0, minimum=1)
    nxt = sim.step(st, {"a": "right"})
    assert nxt["food"] == {(0, 0)}


def test_spawn_chance_100_adds_one_food():
    st = state([snake("a", [(5, 5), (5, 6), (5, 7)])], food=[(0, 0)], chance=100, minimum=1)
    nxt = sim.step(st, {"a": "right"})
    assert len(nxt["food"]) == 2


def test_food_never_spawns_on_snakes_or_existing_food():
    rng_seeds = range(20)
    for seed in rng_seeds:
        st = state([snake("a", [(5, 5), (5, 6), (5, 7)])], food=[(0, 0)], chance=100, seed=seed)
        nxt = sim.step(st, {"a": "right"})
        new_food = nxt["food"] - {(0, 0)}
        (cell,) = new_food
        assert cell not in {(6, 5), (5, 5), (5, 6)}


# --- game over / winner --------------------------------------------------------------


def test_game_over_and_winner_multiplayer():
    st = state(
        [snake("a", [(4, 5), (3, 5), (2, 5)]), snake("b", [(6, 5), (7, 5)])],
        **NO_FOOD,
    )
    assert not sim.game_over(st)
    nxt = sim.step(st, {"a": "right", "b": "left"})  # h2h, b shorter -> dies
    assert sim.game_over(nxt)
    assert sim.winner(nxt) == "a"


def test_game_over_draw_returns_none():
    st = state(
        [snake("a", [(4, 5), (3, 5), (2, 5)]), snake("b", [(6, 5), (7, 5), (8, 5)])],
        **NO_FOOD,
    )
    nxt = sim.step(st, {"a": "right", "b": "left"})
    assert sim.game_over(nxt) and sim.winner(nxt) is None


def test_solo_game_over_when_dead():
    st = state([snake("a", [(10, 5), (9, 5), (8, 5)])], **NO_FOOD)
    nxt = sim.step(st, {"a": "right"})
    assert sim.game_over(nxt) and sim.winner(nxt) is None


# --- init_game -----------------------------------------------------------------------


def test_init_game_official_shape():
    st = sim.init_game(seed=123)
    assert st["width"] == 11 and st["height"] == 11 and st["turn"] == 0
    assert len(st["snakes"]) == 4
    positions = set()
    for s in st["snakes"]:
        assert s["health"] == 100 and s["alive"]
        assert len(s["body"]) == 3
        assert len(set(s["body"])) == 1  # stacked on one cell
        positions.add(s["body"][0])
    assert len(positions) == 4  # distinct starts
    assert len(st["food"]) == 5  # one per snake + center
    assert (5, 5) in st["food"]
    for f in st["food"]:
        assert 0 <= f[0] < 11 and 0 <= f[1] < 11
        assert f not in positions


def test_init_game_reproducible_by_seed():
    a, b = sim.init_game(seed=7), sim.init_game(seed=7)
    assert [s["body"] for s in a["snakes"]] == [s["body"] for s in b["snakes"]]
    assert a["food"] == b["food"]
    assert sim.init_game(seed=8)["food"] != a["food"] or [
        s["body"] for s in sim.init_game(seed=8)["snakes"]
    ] != [s["body"] for s in a["snakes"]]


# --- make_game_state (API view) -------------------------------------------------------


def test_make_game_state_matches_api_schema():
    st = state(
        [snake("a", [(5, 5), (5, 6), (5, 7)], health=77), snake("b", [(1, 1), (1, 2)])],
        food=[(0, 0)],
        turn=42,
    )
    gs = sim.make_game_state(st, "a")
    assert gs["turn"] == 42
    assert gs["board"]["width"] == 11 and gs["board"]["height"] == 11
    assert gs["board"]["food"] == [{"x": 0, "y": 0}]
    assert gs["you"]["id"] == "a"
    assert gs["you"]["health"] == 77
    assert gs["you"]["length"] == 3 == len(gs["you"]["body"])
    assert gs["you"]["head"] == {"x": 5, "y": 5} == gs["you"]["body"][0]
    ids = {s["id"] for s in gs["board"]["snakes"]}
    assert ids == {"a", "b"}
    settings = gs["game"]["ruleset"]["settings"]
    assert settings["foodSpawnChance"] == 15 and settings["minimumFood"] == 1


def test_make_game_state_excludes_dead_snakes():
    dead = snake("d", [(1, 1), (1, 2)])
    dead["alive"] = False
    st = state([snake("a", [(5, 5), (5, 6), (5, 7)]), dead])
    gs = sim.make_game_state(st, "a")
    assert {s["id"] for s in gs["board"]["snakes"]} == {"a"}


# --- property tests: invariants over random rollouts -----------------------------------


def _random_legalish_move(gs, rng):
    from src_rl.core import DIRECTIONS, in_bounds, occupied_cells

    head = (gs["you"]["head"]["x"], gs["you"]["head"]["y"])
    occ = occupied_cells(gs["board"]["snakes"])
    w, h = gs["board"]["width"], gs["board"]["height"]
    options = [
        m
        for m, (dx, dy) in DIRECTIONS.items()
        if in_bounds((head[0] + dx, head[1] + dy), w, h) and (head[0] + dx, head[1] + dy) not in occ
    ]
    return rng.choice(options) if options else "up"


def test_invariants_over_random_rollouts():
    for seed in range(10):
        rng = random.Random(seed)
        st = sim.init_game(seed=seed)
        prev_alive = 4
        for _ in range(150):
            if sim.game_over(st):
                break
            moves = {
                s["id"]: _random_legalish_move(sim.make_game_state(st, s["id"]), rng)
                for s in st["snakes"]
                if s["alive"]
            }
            st = sim.step(st, moves)

            alive = [s for s in st["snakes"] if s["alive"]]
            assert len(alive) <= prev_alive
            prev_alive = len(alive)

            cells = {}
            for s in alive:
                assert 0 <= s["health"] <= 100
                head = s["body"][0]
                assert 0 <= head[0] < st["width"] and 0 <= head[1] < st["height"]
                for c in s["body"]:
                    owner = cells.get(c)
                    assert owner is None or owner == s["id"]  # stacked own tail is fine
                    cells[c] = s["id"]
            for f in st["food"]:
                assert f not in cells
