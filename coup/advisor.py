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

    threshold = _threshold_for_claim(
        claim_context,
        style=style,
        public_state=public_state,
        perspective=claimant,
    )
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


def _threshold_for_claim(
    claim_context,
    *,
    style="Balanced",
    public_state=None,
    perspective: str | None = None,
):
    """
    Compute the p_truth threshold above which we do NOT challenge.
    Lower threshold = more willing to challenge (need less certainty).
    Higher threshold = less willing to challenge (need more certainty).

    Adjustments applied on top of a base value:
    - Style delta (conservative/balanced/aggressive)
    - Game pressure: fewer alive players → +delta (more aggressive)
    - Vulnerability: perspective player has 1 influence → -delta (more cautious)
    - Opponent threat: any opponent at 7+ coins → +delta (time pressure)
    """
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

    # Game-state adjustments
    if public_state is not None:
        alive_players = [
            name for name, state in public_state.players.items()
            if int(state.influence_alive) > 0
        ]
        num_alive = len(alive_players)

        # Fewer players → raise threshold (more aggressive challenges)
        if num_alive <= 2:
            adjusted += 0.10
        elif num_alive <= 3:
            adjusted += 0.05

        # Perspective player's vulnerability
        if perspective is not None and perspective in public_state.players:
            my_influence = int(public_state.players[perspective].influence_alive)
            if my_influence == 1:
                # One life left: be conservative, require stronger evidence before challenging
                adjusted -= 0.08

        # Opponent threat pressure: if any opponent can Coup next turn
        for name in alive_players:
            if perspective and name == perspective:
                continue
            opp_coins = int(public_state.players[name].coins)
            if opp_coins >= 7:
                # They're already in coup range; raise threshold to challenge more
                adjusted += 0.06
                break  # only apply once
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
