"""Policy specs — picklable descriptions turned into callables inside workers.

Specs (tuples, safe to send through multiprocessing):
  ("legacy",)                  frozen deployed baseline (model + its fallback)
  ("legacy_heuristic",)        frozen greedy heuristic
  ("heuristic",)               new safety-aware greedy fallback
  ("random_safe",)             uniform over hard-safety-filtered moves
  ("linear", (c1, ..., c16))   our pipeline with overridden linear coefficients
  ("model",)                   our pipeline with the current model.json weights
"""

import random
from typing import Callable, Dict, Tuple

from src_rl import policy as rl_policy
from src_rl.baselines import legacy_policy
from src_rl.safety import hard_safety_filter, least_bad_move, legal_moves

PolicySpec = Tuple
Policy = Callable[[Dict], str]


def _random_safe(rng: random.Random) -> Policy:
    def choose(game_state: Dict) -> str:
        legal = legal_moves(game_state)
        if not legal:
            return least_bad_move(game_state)
        return rng.choice(hard_safety_filter(game_state, legal))

    return choose


def _linear(coef) -> Policy:
    model = dict(rl_policy._MODEL, coef=list(coef))
    if len(model["coef"]) != len(model["feature_names"]):
        raise ValueError(
            f"expected {len(model['feature_names'])} coefficients, got {len(model['coef'])}"
        )

    def choose(game_state: Dict) -> str:
        ranked = rl_policy.rank_moves(game_state, model)
        if not ranked:
            return least_bad_move(game_state)
        return ranked[0][2]

    return choose


def build_policy(spec: PolicySpec, rng: random.Random) -> Policy:
    kind = spec[0]
    if kind == "legacy":
        return legacy_policy.choose_move
    if kind == "legacy_heuristic":
        return legacy_policy.choose_move_heuristic
    if kind == "heuristic":
        return rl_policy.choose_move_heuristic
    if kind == "random_safe":
        return _random_safe(rng)
    if kind == "linear":
        return _linear(spec[1])
    if kind == "model":
        return rl_policy.choose_move
    raise ValueError(f"unknown policy spec: {spec!r}")
