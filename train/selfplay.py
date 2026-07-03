"""Self-play harness: seeded, reproducible games + a multiprocessing runner.

play_game() is fully deterministic for a given (specs, seed): the simulator's
food RNG and every stochastic policy derive from the game seed. run_games()
fans a job list across worker processes; jobs carry only picklable specs.

Profile from a terminal (NOT a notebook):
  python -m src_rl.train.selfplay --games 240 --procs 0   # 0 = auto
"""

import argparse
import multiprocessing as mp
import random
import time
import zlib
from typing import Dict, List, Optional, Tuple

from src_rl import sim
from src_rl.train.opponents import PolicySpec, build_policy

MAX_TURNS = 500


def _seed_for(game_seed: int, snake_id: str) -> int:
    return zlib.crc32(f"{game_seed}:{snake_id}".encode())


def play_game(
    specs: Dict[str, PolicySpec],
    seed: int,
    width: int = 11,
    height: int = 11,
    max_turns: int = MAX_TURNS,
) -> Dict:
    """Run one game; snake ids are the dict keys. Returns a result record."""
    ids = list(specs)
    state = sim.init_game(width=width, height=height, n_snakes=len(ids), seed=seed, snake_ids=ids)
    policies = {
        sid: build_policy(spec, random.Random(_seed_for(seed, sid))) for sid, spec in specs.items()
    }

    while not sim.game_over(state) and state["turn"] < max_turns:
        moves = {
            s["id"]: policies[s["id"]](sim.make_game_state(state, s["id"]))
            for s in state["snakes"]
            if s["alive"]
        }
        state = sim.step(state, moves)

    # Rank 1 = best. Survivors share rank 1; the earlier you died, the worse.
    def death_turn(s):
        return float("inf") if s["alive"] else s["death_turn"]

    snakes = state["snakes"]
    ranks = {}
    for s in snakes:
        ranks[s["id"]] = 1 + sum(1 for o in snakes if death_turn(o) > death_turn(s))

    return {
        "winner": sim.winner(state),
        "ranks": ranks,
        "turns": state["turn"],
        "death_causes": {s["id"]: s["death_cause"] for s in snakes},
        "survived": {s["id"]: s["alive"] or s["death_turn"] for s in snakes},
        "seed": seed,
    }


Job = Tuple[Dict[str, PolicySpec], int]


def _run_batch(jobs: List[Job]) -> List[Dict]:
    return [play_game(specs, seed) for specs, seed in jobs]


def run_games(jobs: List[Job], processes: Optional[int] = None, batch: int = 8) -> List[Dict]:
    """Run jobs across worker processes (fork), preserving job order."""
    if processes is None:
        processes = max(1, min(120, mp.cpu_count() - 8))
    if processes <= 1 or len(jobs) <= batch:
        return _run_batch(jobs)
    batches = [jobs[i : i + batch] for i in range(0, len(jobs), batch)]
    with mp.get_context("fork").Pool(processes) as pool:
        results = pool.map(_run_batch, batches)
    return [r for chunk in results for r in chunk]


def main() -> None:
    parser = argparse.ArgumentParser(description="profile self-play throughput")
    parser.add_argument("--games", type=int, default=240)
    parser.add_argument("--procs", type=int, default=0, help="0 = auto")
    args = parser.parse_args()

    specs = {
        "cand": ("model",),
        "op1": ("legacy",),
        "op2": ("heuristic",),
        "op3": ("random_safe",),
    }
    jobs: List[Job] = [(specs, seed) for seed in range(args.games)]
    procs = args.procs or None

    t0 = time.perf_counter()
    results = run_games(jobs, processes=procs)
    dt = time.perf_counter() - t0

    wins = sum(1 for r in results if r["winner"] == "cand")
    turns = sum(r["turns"] for r in results) / len(results)
    print(
        f"{len(results)} games in {dt:.1f}s = {len(results) / dt:.1f} games/s | "
        f"cand win-rate {wins / len(results):.3f} | avg turns {turns:.0f}"
    )


if __name__ == "__main__":
    main()
