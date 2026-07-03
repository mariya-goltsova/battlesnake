"""Record fixture games from the official Battlesnake CLI for differential tests.

Run from the repo root: python -m src_rl.tests.record_cli_games
Writes jsonl fixtures to src_rl/tests/fixtures/cli_games/ (pinned CLI version
in src_rl/tools/VERSION). Three settings classes:
  nofood   (chance 0, min 0) — exact food match possible
  minfood  (chance 0, min 1) — feeding covered, food modulo the min-respawn
  normal   (chance 15, min 1) — realistic long games
"""

import subprocess
import time
from pathlib import Path

from src_rl.tests.policy_server import resolve_policy, serve_in_thread

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "src_rl" / "tools" / "battlesnake"
FIXTURES = REPO_ROOT / "src_rl" / "tests" / "fixtures" / "cli_games"

CLASSES = {
    "nofood": ["--foodSpawnChance", "0", "--minimumFood", "0"],
    "minfood": ["--foodSpawnChance", "0", "--minimumFood", "1"],
    "normal": ["--foodSpawnChance", "15", "--minimumFood", "1"],
}
PORTS = (8151, 8152)


def record(n_snakes: int, cls: str, seed: int) -> Path:
    out = FIXTURES / f"{cls}_{n_snakes}p_seed{seed}.jsonl"
    args = [str(CLI), "play", "-W", "11", "-H", "11", "--seed", str(seed), "-o", str(out)]
    for i in range(n_snakes):
        args += ["--name", f"s{i}", "--url", f"http://127.0.0.1:{PORTS[i % 2]}"]
    args += CLASSES[cls]
    subprocess.run(args, check=True, capture_output=True, timeout=120)
    return out


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    servers = [
        serve_in_thread(PORTS[0], resolve_policy("legacy")),
        serve_in_thread(PORTS[1], resolve_policy("legacy_heuristic")),
    ]
    time.sleep(0.5)
    try:
        for cls in CLASSES:
            for n_snakes in (2, 4):
                for seed in (1, 2, 3):
                    path = record(n_snakes, cls, seed)
                    turns = sum(1 for _ in open(path)) - 2
                    print(f"{path.name}: {turns} turns")
    finally:
        for s in servers:
            s.shutdown()


if __name__ == "__main__":
    main()
