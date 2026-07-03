"""Evaluation harness: run policy-vs-policy matches and report metrics.

    python3 arena.py --a algo --b hybrid --games 100 --seed 0 [--json out.json]

Metrics per side: wins/draws, turns survived (avg/median), death causes and
per-move latency percentiles. Start positions alternate every game to remove
positional bias; game i uses seed ``seed + i``.
"""

import argparse
import json
import statistics
from typing import Callable, Dict, List

import logic
import policy as policy_module
from engine import run_game

POLICIES: Dict[str, Callable[[Dict], str]] = {
    "algo": logic.choose_move_heuristic,
    "hybrid": policy_module.choose_move,
}


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def run_match(policy_a: Callable, policy_b: Callable, games: int = 100,
              width: int = 11, height: int = 11, seed: int = 0) -> Dict:
    """Play ``games`` seeded games of A vs B with alternating start positions."""
    wins = {"a": 0, "b": 0, "draws": 0}
    turns: Dict[str, List[int]] = {"a": [], "b": []}
    deaths: Dict[str, Dict[str, int]] = {"a": {}, "b": {}}
    latencies: Dict[str, List[float]] = {"a": [], "b": []}

    for i in range(games):
        # Insertion order decides start cells; swap it every other game.
        if i % 2 == 0:
            players = {"a": policy_a, "b": policy_b}
        else:
            players = {"b": policy_b, "a": policy_a}
        result = run_game(players, width=width, height=height, seed=seed + i)

        if result["winner"] in ("a", "b"):
            wins[result["winner"]] += 1
        else:
            wins["draws"] += 1
        for side in ("a", "b"):
            death = result["deaths"].get(side)
            turns[side].append(death["turn"] if death else result["turns"])
            if death:
                cause = death["cause"]
                deaths[side][cause] = deaths[side].get(cause, 0) + 1
            latencies[side].extend(result["latency_ms"][side])

    return {
        "games": games,
        "wins_a": wins["a"],
        "wins_b": wins["b"],
        "draws": wins["draws"],
        "turns": {
            side: {
                "avg": statistics.fmean(turns[side]),
                "median": statistics.median(turns[side]),
            }
            for side in ("a", "b")
        },
        "deaths": deaths,
        "latency_ms": {
            side: {
                "p50": _percentile(latencies[side], 0.50),
                "p95": _percentile(latencies[side], 0.95),
                "max": max(latencies[side], default=0.0),
            }
            for side in ("a", "b")
        },
    }


def _print_summary(name_a: str, name_b: str, result: Dict) -> None:
    games = result["games"]
    print(f"=== {name_a} (A) vs {name_b} (B): {games} games ===")
    print(f"wins A: {result['wins_a']} ({result['wins_a'] / games:.0%})   "
          f"wins B: {result['wins_b']} ({result['wins_b'] / games:.0%})   "
          f"draws: {result['draws']}")
    for side, name in (("a", name_a), ("b", name_b)):
        t = result["turns"][side]
        lat = result["latency_ms"][side]
        causes = ", ".join(f"{c}={n}" for c, n in sorted(result["deaths"][side].items())) or "none"
        print(f"[{side}] {name}: avg turns {t['avg']:.1f} (median {t['median']:.0f}); "
              f"deaths: {causes}; latency ms p50={lat['p50']:.2f} p95={lat['p95']:.2f} "
              f"max={lat['max']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Battlesnake policy arena")
    parser.add_argument("--a", default="algo", choices=sorted(POLICIES))
    parser.add_argument("--b", default="hybrid", choices=sorted(POLICIES))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", dest="json_path", default=None,
                        help="also write machine-readable results to this file")
    args = parser.parse_args()

    result = run_match(POLICIES[args.a], POLICIES[args.b], games=args.games,
                       width=args.width, height=args.height, seed=args.seed)
    _print_summary(args.a, args.b, result)
    if args.json_path:
        payload = {"policy_a": args.a, "policy_b": args.b, "seed": args.seed, **result}
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"json written to {args.json_path}")


if __name__ == "__main__":
    main()
