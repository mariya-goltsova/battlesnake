"""Serving policy: safety filter -> simulate -> features -> linear scorer -> fallback.

Pipeline per move request:
  1. legal_moves (tail-aware); empty -> least_bad_move
  2. hard_safety_filter (certain-death veto, then losing-h2h veto)
  3. candidate_features on the simulated post-move state for each survivor
  4. standardized linear score from model.json, argmax (tie -> more open space)
Any exception falls back to a safety-aware greedy heuristic, so a model or
feature bug can never crash gameplay.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src_rl.features import HUNGRY_THRESHOLD, build_context, candidate_features
from src_rl.safety import hard_safety_filter, least_bad_move, legal_moves

MODEL_PATH = Path(__file__).with_name("model.json")


def load_model(path: Path = MODEL_PATH) -> Dict:
    with open(path) as fh:
        model = json.load(fh)
    assert len(model["feature_names"]) == len(model["mean"]) == len(model["std"]) == len(model["coef"])
    return model


_MODEL = load_model()


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#10b981",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "rl-0.1.0",
    }


def score_features(feats: Dict[str, float], model: Optional[Dict] = None) -> float:
    """Standardized linear score — the exact function training must reproduce."""
    m = model or _MODEL
    score = m["intercept"]
    for i, name in enumerate(m["feature_names"]):
        std = m["std"][i]
        z = (feats.get(name, 0.0) - m["mean"][i]) / std if std else 0.0
        score += m["coef"][i] * z
    return score


def rank_moves(game_state: Dict, model: Optional[Dict] = None) -> List[Tuple[float, float, str]]:
    """(score, open_space, move) for each surviving move, best first."""
    legal = legal_moves(game_state)
    if not legal:
        return []
    safe = hard_safety_filter(game_state, legal)
    ctx = build_context(game_state)
    ranked = []
    for move in safe:
        feats = candidate_features(game_state, move, ctx)
        ranked.append((score_features(feats, model), feats["open_space"], move))
    ranked.sort(reverse=True)
    return ranked


def choose_move(game_state: Dict) -> str:
    """Return the next move; never raises."""
    try:
        ranked = rank_moves(game_state)
        if not ranked:
            return least_bad_move(game_state)
        return ranked[0][2]
    except Exception:  # noqa: BLE001 - gameplay must survive any policy bug
        return choose_move_heuristic(game_state)


def choose_move_heuristic(game_state: Dict) -> str:
    """Safety-aware greedy fallback: most open space, food pull when hungry."""
    try:
        legal = legal_moves(game_state)
        if not legal:
            return least_bad_move(game_state)
        safe = hard_safety_filter(game_state, legal)
        ctx = build_context(game_state)
        best_move, best_score = safe[0], float("-inf")
        for move in safe:
            feats = candidate_features(game_state, move, ctx)
            score = feats["open_space"]
            if game_state["you"]["health"] < HUNGRY_THRESHOLD and feats["food_reachable"]:
                score += feats["food_score"]
            if score > best_score:
                best_move, best_score = move, score
        return best_move
    except Exception:  # noqa: BLE001 - last resort
        return "up"
