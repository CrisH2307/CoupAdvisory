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
    remaining_copies = int(claim_context.remaining_copies)
    p_truth = _hypergeometric_p_truth(p_role_c, hand_size, remaining_copies)

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


def _hypergeometric_p_truth(p_role: float, hand_size: int, remaining_copies: int, deck_size: int = 15) -> float:
    """
    Compute P(player has at least one copy of the role in hand) using the
    hypergeometric distribution, which correctly models sampling without
    replacement from a finite deck.

    Args:
        p_role: The current belief probability for this role (used to derive
                an effective 'remaining copies in unobserved deck').
        hand_size: Number of cards the player currently holds (influence_alive).
        remaining_copies: How many copies of this role remain in the game
                          (not yet revealed as dead).
        deck_size: Approximate total unobserved cards in circulation.
                   Default 15 = 5 roles × 3 copies.

    Returns:
        Probability in [0, 1] that the player holds at least one of this role.
    """
    from math import comb

    remaining_copies = max(0, int(remaining_copies))
    hand_size = max(1, int(hand_size))
    deck_size = max(hand_size, int(deck_size))

    # If no copies remain, claim is impossible
    if remaining_copies <= 0:
        return 0.0

    # If more copies remain than non-copies, probability is 1
    non_copies = deck_size - remaining_copies
    if non_copies < 0:
        return 1.0

    # P(player has 0 copies) = C(non_copies, hand_size) / C(deck_size, hand_size)
    if hand_size > non_copies:
        # Can't draw hand_size cards from non_copies without replacement
        p_zero = 0.0
    elif hand_size > deck_size:
        p_zero = 0.0
    else:
        numerator = comb(non_copies, hand_size)
        denominator = comb(deck_size, hand_size)
        p_zero = numerator / denominator if denominator > 0 else 0.0

    # Blend with the belief-derived estimate for robustness when deck_size
    # approximation is inaccurate (low information games).
    # Weight: 0.7 hypergeometric + 0.3 belief-derived (independence formula)
    p_hyper = 1.0 - p_zero
    p_independence = 1.0 - (1.0 - _clamp(p_role)) ** hand_size
    p_blended = 0.7 * p_hyper + 0.3 * p_independence

    return _clamp(p_blended)
