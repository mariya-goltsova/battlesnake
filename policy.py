"""The hybrid move-selection pipeline.

    candidates -> hard safety filter -> simulate -> features
               -> linear model scorer -> tie-break -> fallback

Hard rules (safety.py) guarantee we never pick a certain death while an
alternative exists; the model only ranks the surviving candidates. When
nothing is safe, ``least_deadly`` picks the move with the most open space.
"""

from typing import Dict

from core import DIRECTIONS, flood_fill, in_bounds, snake_body
from features import candidate_features
from safety import hard_safety_filter, legal_moves
from sim import blocked_for_legality

# Embedded standardized linear model (means/stds/coefs from the ml baseline).
MODEL: Dict = {
    "feature_names": [
        "space_capped",
        "open_space",
        "voronoi",
        "reaches_tail",
        "escape",
        "h2h_danger",
        "near_bigger_head",
        "near_enemy_head",
        "wall_dist",
        "food_score",
        "food_delta",
        "is_food",
        "dist_to_center",
    ],
    "mean": [
        7.357954545454546,
        100.9034090909091,
        48.26988636363637,
        0.9943181818181818,
        2.4431818181818183,
        0.04261363636363636,
        9.673295454545455,
        4.676136363636363,
        1.625,
        0.8920454545454546,
        0.14772727272727273,
        0.036931818181818184,
        5.056818181818182,
    ],
    "std": [
        3.5995966185276513,
        22.80542174802676,
        31.41119158524981,
        0.07516338951888041,
        0.6235520417417705,
        0.20198444088469822,
        7.9675173248507924,
        2.2532045017839604,
        1.3552297691803878,
        5.861056404757769,
        0.9449599886584031,
        0.18859442989548575,
        2.34451950177747,
    ],
    "coef": [
        0.00010539398521136327,
        -1.6778512168946185,
        80.89420182766183,
        9.793855564450467,
        0.7884630868036275,
        -11.025170822665032,
        -0.7981723553489,
        0.5410534990053248,
        1.5629078731518526,
        7.582325762611304,
        0.12463070008097832,
        0.21036618806863483,
        1.836259515524985,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,
}


def _model_score(feats: Dict[str, float]) -> float:
    """Standardized linear score: intercept + sum(coef * z)."""
    score = MODEL["intercept"]
    for i, name in enumerate(MODEL["feature_names"]):
        std = MODEL["std"][i]
        z = (feats.get(name, 0.0) - MODEL["mean"][i]) / std if std else 0.0
        score += MODEL["coef"][i] * z
    return score


def least_deadly(state: Dict) -> str:
    """Best-effort move when no safe candidate exists.

    Prefer legal-but-contested moves over moves into bodies; within the pool,
    take the one with the most reachable space. Last resort: "up".
    """
    board = state["board"]
    width, height = board["width"], board["height"]
    head = snake_body(state["you"])[0]

    pool = legal_moves(state)
    if not pool:
        pool = [
            move
            for move, (dx, dy) in DIRECTIONS.items()
            if in_bounds((head[0] + dx, head[1] + dy), width, height)
        ]
    if not pool:
        return "up"

    blocked = blocked_for_legality(state)

    def space(move: str) -> int:
        dx, dy = DIRECTIONS[move]
        cell = (head[0] + dx, head[1] + dy)
        return flood_fill(cell, blocked - {cell}, width, height)

    return max(pool, key=space)


def choose_move(state: Dict) -> str:
    """Return the best move via safety filter + model ranking."""
    candidates = hard_safety_filter(state)
    if not candidates:
        return least_deadly(state)

    best_move = candidates[0]
    best_score = float("-inf")
    for move in candidates:
        score = _model_score(candidate_features(state, move))
        if score > best_score:  # ties keep the earlier (filter-ordered) move
            best_score = score
            best_move = move
    return best_move
