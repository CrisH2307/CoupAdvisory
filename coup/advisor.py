from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ClaimContext(BaseModel):
    action_name: str | None = None
    block_type: str | None = None
    remaining_copies: int


class Recommendation(BaseModel):
    recommendation: str
    explanation: str
    p_truth: float
    threshold: float


def recommend_challenge(
    belief_state,
    claimant,
    claimed_role,
    claim_context,
    *,
    style: Literal["Conservative", "Balanced", "Aggressive"] | str = "Balanced",
    public_state=None,
    hand_size: int | None = None,
):
    player_probs = belief_state.probabilities.get(claimant, {})
    p_role = float(player_probs.get(claimed_role, 0.0))

    if hand_size is None and public_state is not None:
        player_state = public_state.players.get(claimant)
        if player_state is not None:
            hand_size = int(player_state.influence_alive)
    if hand_size is None or hand_size < 1:
        hand_size = 1
    p_role_c = _clamp(p_role, 0.0, 1.0)
    p_truth = 1.0 - (1.0 - p_role_c) ** hand_size

    threshold = _threshold_for_claim(claim_context, style=style)
    if p_truth < threshold:
        recommendation = "Challenge"
    else:
        recommendation = "Do not challenge"

    remaining = int(claim_context.remaining_copies)
    normalized_style = _normalize_style(style)
    explanation = (
        f"style={normalized_style}, p_truth={p_truth:.2f} vs threshold={threshold:.2f} "
        f"(p_role={p_role:.2f}, hand={hand_size}, remaining={remaining})"
    )
    return Recommendation(
        recommendation=recommendation,
        explanation=explanation,
        p_truth=p_truth,
        threshold=threshold,
    )


def _threshold_for_claim(claim_context, *, style="Balanced"):
    remaining = int(claim_context.remaining_copies)
    if remaining <= 0:
        base = 0.99
    elif claim_context.block_type == "Steal":
        base = 0.20
    elif claim_context.action_name == "Assassinate":
        base = 0.35
    elif claim_context.block_type == "Assassination":
        base = 0.35
    else:
        base = 0.25

    adjusted = float(base) + _style_threshold_delta(style)
    return _clamp(adjusted, 0.01, 0.99)


def _normalize_style(style):
    value = str(style).strip().lower()
    if value == "conservative":
        return "Conservative"
    if value == "aggressive":
        return "Aggressive"
    return "Balanced"


def _style_threshold_delta(style):
    normalized = _normalize_style(style)
    if normalized == "Conservative":
        return -0.05
    if normalized == "Aggressive":
        return 0.07
    return 0.0


def _clamp(value, low=0.0, high=1.0):
    if value < low:
        return float(low)
    if value > high:
        return float(high)
    return float(value)
