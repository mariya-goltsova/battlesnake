# Stage 0: Shared Hybrid Core (from unite_algo_ml.txt)

Цель: общее ядро (core/sim/safety/features/policy), P0-фичи (simulate_after_move,
BFS-еда, h2h-veto), тесты и eval-харнесс. После этого этапа гибрид собран и измерим.

## Global Constraints

- Python 3.11, **stdlib only** in game logic (no numpy/pandas/sklearn in the serve path).
  Flask/gunicorn stay only in `backend.py`.
- Board coords: `(0,0)` is bottom-left; `up -> y+1`, `down -> y-1`, `left -> x-1`, `right -> x+1`.
- Game-state dicts follow the Battlesnake API schema exactly (https://docs.battlesnake.com/api):
  `state = {"turn", "board": {"width", "height", "food": [{"x","y"}...], "snakes": [...]}, "you": {...}}`,
  snake = `{"id", "health", "length", "head": {"x","y"}, "body": [{"x","y"}, ...]}` (body[0] == head).
- Public API must keep working unchanged: `from logic import choose_move, get_info`;
  `backend.py` is NOT modified.
- The 13 model feature names in `_MODEL["feature_names"]` must remain unchanged
  (extra new feature keys are allowed — the scorer iterates over model names only).
- Latency: `choose_move` on 11x11 must stay well under 100ms (a handful of BFS over
  121 cells per candidate is fine; no per-move loops over all cell pairs).
- Battlesnake body subtlety (MUST be encoded in sim + engine + tests): when a snake
  eats, the engine appends a duplicate tail segment, so `body[-1] == body[-2]` for one
  turn and that tail cell does NOT free on the next move. A tail cell frees on a tick
  only if `body[-1] != body[-2]` and the snake does not eat this tick.
- Tests: pytest, files under `tests/`. Run with `python3 -m pytest tests/ -q`.
- Git hygiene: `git add` only the files of your task, never `git add -A` /`git add .`
  (repo has unrelated untracked files). Commit messages in English, imperative.
- All new module-level functions get short docstrings; match the existing code style
  of `logic.py` (typing via `Dict/List/Set/Tuple`, `Point = Tuple[int, int]`).

## Task 1: Extract shared primitives into core.py

**Files:** `core.py` (new), `logic.py` (edit), `tests/test_core.py` (new)

Create `core.py` with the shared primitives, moved verbatim (public names, no leading
underscore) from `logic.py`:

- `Point`, `DIRECTIONS` (dict move->delta, insertion order up/down/left/right)
- `in_bounds(p, width, height) -> bool`
- `manhattan(a, b) -> int`
- `bfs_dist(sources, blocked, width, height) -> Dict[Point, int]` — from `_bfs_dist`
  (logic.py:176). Note: sources are seeded at distance 0 even if listed in `blocked`;
  neighbors are only expanded into cells not in `blocked`.
- `flood_fill(start, occupied, width, height, limit=None) -> int` — from `_flood_fill`
  (logic.py:136), but `limit=None` means uncapped (bounded by board size).
- `occupied_cells(snakes) -> Set[Point]` — from `_occupied_cells` (all segments, tails included).
- `head_to_head_cells(snakes, my_id, my_length) -> Set[Point]` — from `_head_to_head_cells`:
  cells adjacent to heads of enemies with `length >= my_length`.
- `snake_body(snake) -> List[Point]` helper: `[(seg["x"], seg["y"]) for seg in snake["body"]]`.
- `food_points(board) -> List[Point]`.

Update `logic.py` to import these from `core` and delete its local copies; keep thin
underscore aliases if needed so existing internal call sites still work. Behavior must
be bit-for-bit identical (this is a pure refactor).

**Tests (`tests/test_core.py`):**
- `bfs_dist`: distances on an empty 5x5 from (0,0); a wall of blocked cells splits the
  board — far side unreachable (absent from dict).
- `flood_fill`: uncapped equals free-cell count of a small pocket; capped stops at limit.
- `in_bounds` edge cells; `manhattan` symmetry.
- `occupied_cells` / `head_to_head_cells` on a tiny 2-snake fixture: shorter enemy
  contributes no danger cells, equal/longer does.
- Regression: `from logic import choose_move` still returns a legal move on a simple
  11x11 one-snake state (build the state dict inline).

**Verify:** `python3 -m pytest tests/ -q` green; `python3 -c "import logic, backend"` ok.
Commit.

## Task 2: Next-state simulation in sim.py

**Files:** `sim.py` (new), `tests/test_sim.py` (new)

`sim.py` implements the corrected occupancy model (P0-1 of both plans):

```python
def simulate_after_move(state: Dict, move: str) -> Dict:
    """Simulate our snake playing `move`. Returns:
    {
      "next_pos": Point,          # our head after the move
      "ate": bool,                # next_pos was food
      "my_body": List[Point],     # our body after the move (head first)
      "occupied": Set[Point],     # all cells occupied after our move resolves
    }
    """
```

Rules to implement:
- `next_pos = head + DIRECTIONS[move]`; `ate = next_pos in food`.
- Our body after move: `[next_pos] + body[:-1]` if not `ate`, else `[next_pos] + body`
  (growth keeps the tail).
- Own stacked tail: if `body[-1] == body[-2]`, the tail cell stays occupied even when
  not eating (set construction over remaining segments handles this naturally —
  document it and test it).
- Enemy bodies: keep all segments, but free an enemy's tail cell if
  `body[-1] != body[-2]` AND no food is adjacent to that enemy's head (conservative:
  if the enemy *could* eat this tick, assume its tail stays).
- Enemy heads do not move in this model (their possible next cells are handled as
  h2h danger by safety.py, not as occupancy).
- `occupied` = our simulated body cells ∪ enemy cells after tail handling.
- Also export `blocked_for_legality(state) -> Set[Point]`: current occupancy for
  *choosing* a move — all segments of all snakes, minus our own tail if
  `body[-1] != body[-2]`, minus each enemy tail if `body[-1] != body[-2]` and no food
  adjacent to that enemy's head. (Moving onto a tail cell that is about to free is
  legal; food can never be under a body, so stepping on our own tail never feeds us.)

**Tests (`tests/test_sim.py`):** build small explicit states:
- not eating: own tail cell freed, body shifted, length constant;
- eating: tail kept, body grows by 1, `ate` is True;
- stacked tail (`body[-1]==body[-2]`): tail cell NOT freed when not eating;
- enemy with food adjacent to its head: enemy tail NOT freed; without food: freed;
- `blocked_for_legality`: own tail cell is legal to enter (not blocked) in the plain
  case, blocked in the stacked case.

**Verify:** pytest green. Commit.

## Task 3: Hard safety filter in safety.py (h2h veto)

**Files:** `safety.py` (new), `tests/test_safety.py` (new)

`safety.py` (P0-3 of both plans), built on `core` + `sim`:

```python
def legal_moves(state: Dict) -> List[str]:
    """Moves that stay in bounds and don't enter a blocked cell
    (uses sim.blocked_for_legality, i.e. tail-aware)."""

def hard_safety_filter(state: Dict) -> List[str]:
    """legal_moves minus h2h-losing cells — but the veto only applies while at
    least one non-vetoed move remains. Returns possibly-empty list; policy
    handles the empty case with a least-deadly fallback."""
```

- h2h veto: candidate cell in `core.head_to_head_cells(...)` (adjacent to an enemy
  head with `length >= my_length`) is dropped iff at least one legal candidate is
  outside that set. If ALL legal candidates are vetoed, return them all (policy will
  pick among them), ordered so that cells contested only by *equal*-length enemies come
  before cells contested by strictly longer ones (a tie is better than a loss).

**Tests (`tests/test_safety.py`)** — the canonical board states (algo plan P2-1):
1. wall: head in a corner — only in-bounds moves are legal;
2. own tail: moving onto own (non-stacked) tail is legal;
3. stacked tail: same cell illegal right after eating;
4. h2h vs longer enemy: contested cell dropped when an alternative exists;
5. h2h vs shorter enemy: contested cell NOT dropped;
6. all-moves-contested: filter returns the full legal list (no empty-out), equal-length
   contested cells ordered before longer-enemy cells;
7. dead end vs open: both stay in the list (space choice is scoring's job, not safety's) —
   documents the layer boundary;
8. fully trapped (no legal moves): returns `[]`.

**Verify:** pytest green. Commit.

## Task 4: Shared features in features.py (BFS food)

**Files:** `features.py` (new), `logic.py` (edit: drop `_candidate_features`),
`tests/test_features.py` (new)

Move feature computation into a single shared module (P0-2 + train/serve parity):

```python
def candidate_features(state: Dict, move: str) -> Dict[str, float]:
    """Feature vector for playing `move`. Assumes the move is legal.
    Computed on the simulated next state (sim.simulate_after_move)."""
```

Keep all 13 existing feature names with these corrected computations (let
`simres = simulate_after_move(state, move)`, `blocked = simres["occupied"] - {next_pos}`):

- `space_capped`, `open_space`: `flood_fill(next_pos, blocked, ...)` capped at
  `my_length + 1` / uncapped;
- `voronoi`: `bfs_dist([next_pos], blocked, ...)` vs `bfs_dist(enemy_heads, blocked, ...)`;
  count cells we reach strictly first;
- `reaches_tail`: our NEW tail = `simres["my_body"][-1]`; reachable via
  `bfs_dist([next_pos], blocked - {tail}, ...)`;
- `escape`: free in-bounds neighbors of `next_pos` on the simulated map;
- `h2h_danger`, `near_bigger_head`, `near_enemy_head`: same semantics as now
  (manhattan to heads is fine — proximity, not path);
- `wall_dist`, `dist_to_center`: unchanged;
- food block — **BFS instead of Manhattan**:
  - `bfs_food_next` = min BFS distance from `next_pos` to any food over `blocked`
    (0 if `ate`); `bfs_food_now` = min BFS distance from the current head on the
    current map (`core.occupied_cells` minus own head);
  - unreachable food -> distance `BIG = 10_000`;
  - `food_score = (W + H - bfs_food_next) * 2` if hungry (`health < 50`) and reachable
    else `0.0`;
  - `food_delta = bfs_food_now - bfs_food_next` if any food reachable from either
    point else `0.0` (clamp to `[-(W+H), W+H]` to avoid BIG leaking into the value);
  - `is_food = 1.0 if ate`.
- New extra keys (safe — scorer only reads model names): `bfs_food_dist`
  (= `bfs_food_next`, BIG if unreachable), `food_reachable` (0/1).

`logic.py`: `_candidate_features` is deleted; `choose_move_model` imports
`candidate_features` from `features`. Behavior of the heuristic fallback is untouched.

**Tests (`tests/test_features.py`):**
- food behind a body wall: manhattan-close but BFS-unreachable -> `food_reachable == 0`,
  `food_score == 0`;
- tail-freed corridor: a path that only opens because a tail frees (sim-aware map)
  yields finite `bfs_food_dist`;
- `is_food`/`ate` consistency; `reaches_tail` true when following own tail;
- all 13 model names present in the returned dict, all values floats.

**Verify:** pytest green; `python3 -c "import logic"` ok. Commit.

## Task 5: Pipeline in policy.py, logic.py becomes a thin wrapper

**Files:** `policy.py` (new), `logic.py` (edit), `tests/test_policy.py` (new)

`policy.py` implements the target pipeline from unite_algo_ml.txt:

```python
def choose_move(state: Dict) -> str:
    """hard_safety_filter -> features -> linear scorer -> tie-break -> fallback."""
```

- Move the `_MODEL` dict from `logic.py` into `policy.py` (unchanged values) with the
  same pure-python standardized scoring (score = Σ coef·(x-mean)/std).
- Pipeline: `candidates = hard_safety_filter(state)`;
  - if empty -> `least_deadly(state)`: among ALL in-bounds moves (even into bodies /
    vetoed cells), prefer legal-but-vetoed over body cells, then max uncapped
    `flood_fill`; final default `"up"`;
  - else score each candidate with the model over `features.candidate_features`,
    argmax; on exact ties keep the earlier candidate (stable order from the filter).
- `logic.py` after this task: keeps `get_info`, `choose_move_heuristic` (unchanged,
  used as last-resort fallback), and
  `choose_move = try policy.choose_move except -> choose_move_heuristic`.
  `choose_move_model` is deleted (superseded by the pipeline).

**Tests (`tests/test_policy.py`):**
- end-to-end on inline fixtures: returns a legal move on an open board;
- h2h veto respected end-to-end (contested cell not chosen when alternative exists);
- fully trapped -> still returns one of the 4 move strings (no exception);
- model scoring: with a hand-built 2-candidate state, the returned move matches an
  independent recomputation of the linear score in the test;
- `from logic import choose_move` smoke: still works, exceptions in policy fall back
  to the heuristic (monkeypatch policy.choose_move to raise; expect a legal move).

**Verify:** pytest green (whole suite); `python3 -c "import backend"` ok. Commit.

## Task 6: Minimal rules engine in engine.py

**Files:** `engine.py` (new), `tests/test_engine.py` (new)

Pure-python Battlesnake standard-rules engine for local games (no HTTP). This powers
the arena (Task 7) and later self-play data generation (Stage 2). Seeded and
deterministic.

```python
def run_game(policies: Dict[str, Callable[[Dict], str]], width: int = 11,
             height: int = 11, seed: int = 0, max_turns: int = 1000) -> Dict:
    """Play one game. Returns:
    {"winner": snake_id or None (draw), "turns": int,
     "deaths": {snake_id: {"turn": int, "cause": str}},
     "latency_ms": {snake_id: List[float]}}   # per-move policy latency
    """
```

Standard rules, applied each turn in this order (reference: official rules repo):
1. gather moves: call each live snake's policy with an API-schema state where `you`
   is that snake (deep-ish copy so policies can't mutate engine state); a policy
   exception or illegal string counts as `"up"`;
2. move snakes simultaneously: new head = head + delta; remove tail segment;
3. health -= 1;
4. feed: if head is on food — remove that food, health = 100, append duplicate tail
   segment (this creates the stacked tail; length += 1);
5. spawn food with probability 15% per turn on a uniformly random free cell, and
   always top up to at least 1 food on the board (`random.Random(seed)`);
6. eliminations (simultaneous, based on post-move positions):
   - out of bounds -> cause `"wall"`;
   - health <= 0 -> cause `"starvation"`;
   - head in any snake's body segment (its own body[1:] -> `"self-collision"`, another
     snake's body[1:] -> `"body-collision"`);
   - head-to-head (two heads on the same cell): the shorter snake dies (`"head-to-head"`);
     equal lengths -> both die.
7. game ends when <=1 snake remains (winner = survivor or None) or `max_turns` reached
   (winner = longest alive snake, tie -> None).

Initial position: each snake length 3, all segments stacked on its start cell (this is
how the official engine starts — first moves naturally unstack); start cells for 2
snakes at `(1,1)` and `(width-2, height-2)`; 1 initial food at a seeded free cell.

Build the per-snake state dict in the exact API schema (see Global Constraints) —
`turn`, `board.width/height/food/snakes`, `you`; each snake carries `id`, `health`,
`length`, `head`, `body`.

**Tests (`tests/test_engine.py`):** scripted policies (deterministic move sequences):
- snake moving up on an empty board eventually dies on the `"wall"` at the right turn;
- eating: food consumed, length +1, health 100, stacked tail visible for exactly one
  turn (`body[-1]==body[-2]`), tail cell blocked that turn;
- starvation at health 0;
- body-collision and self-collision causes;
- head-to-head: longer survives; equal -> both die, game ends with winner None;
- determinism: same seed + same scripted policies -> identical result dict.

**Verify:** pytest green. Commit.

## Task 7: Arena harness in arena.py + smoke run

**Files:** `arena.py` (new), `tests/test_arena.py` (new)

CLI harness over `engine.run_game` (P2-2 of the algo plan):

```
python3 arena.py --a algo --b hybrid --games 100 --width 11 --height 11 --seed 0 [--json out.json]
```

- Policy registry: `"algo" -> logic.choose_move_heuristic`, `"hybrid" -> policy.choose_move`.
- Runs N games alternating start positions (swap who starts where each game to remove
  positional bias); per game seed = `seed + game_index`.
- Prints a human-readable summary and (with `--json`) writes machine-readable results:
  - wins A / wins B / draws, win-rate with the count;
  - avg and median turns survived per policy;
  - death causes per policy (wall / body-collision / self-collision / head-to-head /
    starvation counts);
  - per-move latency per policy: p50 / p95 / max (ms).
- Keep it importable: `run_match(policy_a, policy_b, games, width, height, seed) -> dict`
  with the CLI as a thin wrapper.

**Tests (`tests/test_arena.py`):** `run_match` with two trivial scripted policies for
2 games returns the right structure (keys, counts sum to games); registry contains
`algo` and `hybrid`.

**Verify:** pytest green, then the smoke run (report its output in your report):
`python3 arena.py --a algo --b hybrid --games 30 --seed 0`. It must complete without
errors; win-rates are informational (calibrating the hybrid is Stage 1). Commit.
