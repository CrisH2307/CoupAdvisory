from __future__ import annotations

from coup.models import (
    ActionEvent,
    BlockEvent,
    ChallengeEvent,
    InfluenceLostEvent,
    RevealEvent,
    Role,
)
from coup.rules import ACTION_ROLE_MULTIPLIERS, BLOCK_MULTIPLIERS

# ruff: noqa: E501


def apply_event(belief, public_state, event, *, strict_no_duplicate_hand=False):
    if isinstance(event, ActionEvent):
        action = ACTION_ROLE_MULTIPLIERS.get(event.action_name)
        if action:
            role, factor = action
            belief.scores[event.actor][role] *= factor
    elif isinstance(event, BlockEvent):
        for role, factor in BLOCK_MULTIPLIERS.get(event.blocked_action, []):
            belief.scores[event.blocker][role] *= factor
    elif isinstance(event, ChallengeEvent):
        factor = 10.0 if event.result == "win" else 0.01
        belief.scores[event.challenged][event.claimed_role] *= factor

    _apply_deck_pressure(belief, public_state)
    _normalize(belief)
    _apply_reveal_constraints(
        belief,
        public_state,
        event,
        strict_no_duplicate_hand=strict_no_duplicate_hand,
    )


def _apply_deck_pressure(belief, public_state):
    for role in Role:
        remaining = max(0, 3 - public_state.revealed_dead.get(role, 0))
        damp = (remaining / 3.0) ** 0.7 if remaining > 0 else 0.0
        for player in belief.scores:
            belief.scores[player][role] *= damp


def _normalize(belief):
    for player, scores in belief.scores.items():
        total = sum(scores.values())
        if total <= 0:
            belief.probabilities[player] = {role: 1.0 / len(Role) for role in Role}
            continue
        belief.probabilities[player] = {role: score / total for role, score in scores.items()}


def _apply_reveal_constraints(belief, public_state, event, *, strict_no_duplicate_hand=False):
    revealed_by_player: dict[str, Role] = {}
    for past_event in public_state.history:
        if isinstance(past_event, (RevealEvent, InfluenceLostEvent)):
            revealed_by_player[past_event.player] = past_event.revealed_role
    if isinstance(event, (RevealEvent, InfluenceLostEvent)):
        revealed_by_player[event.player] = event.revealed_role

    for player, role in revealed_by_player.items():
        influence = public_state.players[player].influence_alive
        if influence <= 0:
            belief.scores[player] = {r: 0.0 for r in Role}
            belief.probabilities[player] = {r: 0.0 for r in Role}
            continue

        if influence == 1:
            probs = dict(belief.probabilities[player])
            if strict_no_duplicate_hand:
                probs[role] = 0.0
            else:
                # One revealed card means the remaining hidden card is less likely
                # to be the same role, but still possible in standard Coup mode.
                probs[role] *= 0.2
            total = sum(probs.values())
            if total > 0:
                normalized = {r: (probs[r] / total) for r in Role}
            elif strict_no_duplicate_hand:
                fallback = 1.0 / (len(Role) - 1)
                normalized = {r: (0.0 if r == role else fallback) for r in Role}
            else:
                fallback = 1.0 / len(Role)
                normalized = {r: fallback for r in Role}
            belief.probabilities[player] = normalized
            belief.scores[player] = dict(normalized)
