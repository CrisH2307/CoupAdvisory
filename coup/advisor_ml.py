from __future__ import annotations

import os
import pickle
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

from coup.advisor import (
    ClaimContext,
    Recommendation,
    _normalize_style,
    _style_threshold_delta,
    _threshold_for_claim,
)
from coup.models import BeliefState, PublicState, Role

ROLE_ORDER = [Role.DUKE, Role.ASSASSIN, Role.CAPTAIN, Role.AMBASSADOR, Role.CONTESSA]
ACTION_OPTIONS = ["Tax", "Steal", "Assassinate", "Exchange"]
BLOCK_OPTIONS = ["Foreign Aid", "Assassination", "Steal"]

NUMERIC_FEATURES = [
    "f_players_alive",
    "f_total_coins",
    "f_actor_coins",
    "f_actor_influence",
    "f_remaining_duke",
    "f_remaining_assassin",
    "f_remaining_captain",
    "f_remaining_ambassador",
    "f_remaining_contessa",
    "f_belief_p_truth",
    "f_remaining_claimed",
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + [f"role_{r.value.lower()}" for r in ROLE_ORDER]
    + [f"action_{a.lower().replace(' ', '_')}" for a in ACTION_OPTIONS]
    + [f"block_{b.lower().replace(' ', '_')}" for b in BLOCK_OPTIONS]
)

_MODEL_CACHE: dict[str, dict] = {}
_CACHE_LOCK = Lock()
_DEFAULT_MODEL = "runs/bench_heuristic/challenge_head/model.pkl"


def resolve_model_path(model_path: Optional[str] = None) -> Path:
    candidate = model_path or os.environ.get("COUP_ADVISOR_ML_MODEL") or _DEFAULT_MODEL
    return Path(candidate)


def load_model(model_path: Optional[str] = None) -> dict:
    path = resolve_model_path(model_path)
    key = str(path.resolve()) if path.exists() else str(path)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        if not path.exists():
            raise FileNotFoundError(
                f"ML advisor model not found at {path}. Train one with "
                f"`python -m coup.research.train_challenge_head --dataset <csv> --out <dir>`."
            )
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        _MODEL_CACHE[key] = payload
        return payload


def build_feature_row(
    belief_state: BeliefState,
    claimant: str,
    claimed_role: Role,
    claim_context: ClaimContext,
    public_state: Optional[PublicState] = None,
) -> dict[str, float]:
    row = {col: 0.0 for col in FEATURE_COLUMNS}

    players_alive = 0
    total_coins = 0
    actor_coins = 0
    actor_influence = 0
    if public_state is not None:
        players_alive = sum(
            1 for p in public_state.players.values() if p.influence_alive > 0
        )
        total_coins = sum(p.coins for p in public_state.players.values())
        actor = public_state.players.get(claimant)
        if actor is not None:
            actor_coins = int(actor.coins)
            actor_influence = int(actor.influence_alive)

    remaining_map = _remaining_from_belief(belief_state, public_state)
    row["f_players_alive"] = float(players_alive)
    row["f_total_coins"] = float(total_coins)
    row["f_actor_coins"] = float(actor_coins)
    row["f_actor_influence"] = float(actor_influence)
    row["f_remaining_duke"] = float(remaining_map.get(Role.DUKE, 0))
    row["f_remaining_assassin"] = float(remaining_map.get(Role.ASSASSIN, 0))
    row["f_remaining_captain"] = float(remaining_map.get(Role.CAPTAIN, 0))
    row["f_remaining_ambassador"] = float(remaining_map.get(Role.AMBASSADOR, 0))
    row["f_remaining_contessa"] = float(remaining_map.get(Role.CONTESSA, 0))
    row["f_remaining_claimed"] = float(int(claim_context.remaining_copies))

    probs = belief_state.probabilities.get(claimant, {})
    p_truth = 0.0
    for role, value in probs.items():
        if role == claimed_role or getattr(role, "value", role) == getattr(
            claimed_role, "value", claimed_role
        ):
            p_truth = float(value)
            break
    row["f_belief_p_truth"] = p_truth

    role_value = getattr(claimed_role, "value", claimed_role)
    role_key = f"role_{role_value.lower()}"
    if role_key in row:
        row[role_key] = 1.0

    if claim_context.action_name in ACTION_OPTIONS:
        key = f"action_{claim_context.action_name.lower().replace(' ', '_')}"
        if key in row:
            row[key] = 1.0
    if claim_context.block_type in BLOCK_OPTIONS:
        key = f"block_{claim_context.block_type.lower().replace(' ', '_')}"
        if key in row:
            row[key] = 1.0

    return row


def _remaining_from_belief(belief_state: BeliefState, public_state: Optional[PublicState]):
    if public_state is None:
        return {role: 0 for role in Role}
    dead = public_state.revealed_dead
    alive_by_role = {role: 3 for role in Role}
    for role, count in dead.items():
        key = role if isinstance(role, Role) else Role(role)
        alive_by_role[key] = max(0, 3 - int(count))
    return alive_by_role


def recommend_challenge_ml(
    belief_state: BeliefState,
    claimant: str,
    claimed_role: Role,
    claim_context: ClaimContext,
    *,
    style: Literal["Conservative", "Balanced", "Aggressive"] | str = "Balanced",
    model_path: Optional[str] = None,
    public_state: Optional[PublicState] = None,
) -> Recommendation:
    payload = load_model(model_path)
    model = payload["model"]
    feature_cols = payload.get("feature_cols", FEATURE_COLUMNS)
    version = payload.get("version", "ch-head-v1")

    row = build_feature_row(belief_state, claimant, claimed_role, claim_context, public_state)
    x = [[row.get(col, 0.0) for col in feature_cols]]

    classes = [int(c) for c in list(getattr(model, "classes_", [0, 1]))]
    proba = model.predict_proba(x)[0]
    # y=1 means "claim was historically challenged" → bluff-like signal.
    # p_truth = probability this claim will NOT be challenged = p(y=0).
    not_challenged_idx = classes.index(0) if 0 in classes else 0
    p_truth = float(proba[not_challenged_idx])
    p_challenge = 1.0 - p_truth

    threshold = _threshold_for_claim(claim_context, style=style)
    if p_truth < threshold:
        recommendation = "Challenge"
    else:
        recommendation = "Do not challenge"

    normalized_style = _normalize_style(style)
    remaining = int(claim_context.remaining_copies)
    explanation = (
        f"engine=ml[{version}], style={normalized_style}, "
        f"p_truth={p_truth:.2f} (model p_challenge={p_challenge:.2f}) "
        f"vs threshold={threshold:.2f} with remaining={remaining}"
    )
    return Recommendation(
        recommendation=recommendation,
        explanation=explanation,
        p_truth=p_truth,
        threshold=threshold,
    )


def clear_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


# Expose the style delta for callers (ensemble) without reimporting coup.advisor internals.
style_threshold_delta = _style_threshold_delta
