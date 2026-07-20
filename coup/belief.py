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


def apply_event(belief, public_state, event, *, strict_no_duplicate_hand=False, self_pin=None):
    if isinstance(event, ActionEvent):
        _apply_action_multiplier(belief, event)
    elif isinstance(event, BlockEvent):
        _apply_block_multiplier(belief, event)
    elif isinstance(event, ChallengeEvent):
        _apply_challenge_multiplier(belief, event)

    _apply_deck_pressure(belief, public_state)
    if isinstance(event, ActionEvent) and event.action_name == "Exchange":
        _reset_to_deck_prior(belief, public_state, event.actor)
    _normalize(belief)
    _apply_reveal_constraints(
        belief,
        public_state,
        event,
        strict_no_duplicate_hand=strict_no_duplicate_hand,
    )
    if self_pin is not None:
        _apply_self_pin(belief, public_state, self_pin)


def _apply_self_pin(belief, public_state, self_pin):
    player, hand = self_pin
    if player not in belief.scores:
        return
    alive = int(public_state.players[player].influence_alive) if player in public_state.players else len(hand)
    effective = list(hand)[:max(alive, 0)]
    counts = {role: 0 for role in Role}
    for role in effective:
        counts[role] += 1
    total = sum(counts.values())
    if total <= 0:
        probs = {role: 0.0 for role in Role}
    else:
        probs = {role: counts[role] / total for role in Role}
    belief.probabilities[player] = probs
    belief.scores[player] = dict(probs)


def _apply_action_multiplier(belief, event):
    if event.actor not in belief.scores:
        return
    from coup.rules import ACTION_LIKELIHOOD_RATIOS
    entry = ACTION_LIKELIHOOD_RATIOS.get(event.action_name)
    if entry is None:
        return
    role_name, ratio = entry
    # Convert string role name to Role enum for dict key lookup
    from coup.models import Role
    role = Role(role_name)
    belief.scores[event.actor][role] *= float(ratio)


def _apply_block_multiplier(belief, event):
    if event.blocker not in belief.scores:
        return
    from coup.rules import BLOCK_LIKELIHOOD_RATIOS
    from coup.models import Role
    for role_name, ratio in BLOCK_LIKELIHOOD_RATIOS.get(event.blocked_action, []):
        role = Role(role_name)
        belief.scores[event.blocker][role] *= float(ratio)


def _apply_challenge_multiplier(belief, event):
    if event.challenged not in belief.scores:
        return
    from coup.rules import CHALLENGE_WIN_RATIO, CHALLENGE_LOSE_RATIO
    if event.result == "win":
        scale = CHALLENGE_WIN_RATIO
    else:
        scale = CHALLENGE_LOSE_RATIO
    belief.scores[event.challenged][event.claimed_role] *= scale


def _reset_to_deck_prior(belief, public_state, player):
    if player not in belief.scores:
        return
    for role in Role:
        revealed = int(public_state.revealed_dead.get(role, 0))
        remaining = max(0, 3 - revealed)
        belief.scores[player][role] = max(float(remaining), 0.01)
    belief.scores[player][Role.AMBASSADOR] *= 1.5


def _apply_deck_pressure(belief, public_state):
    for role in Role:
        revealed = int(public_state.revealed_dead.get(role, 0))
        remaining = max(0, 3 - revealed)
        if remaining <= 0:
            damp = 0.0
        else:
            damp = (remaining / 3.0) ** 0.7
        for player in belief.scores:
            belief.scores[player][role] *= damp


def _normalize(belief):
    for player in sorted(belief.scores):
        _normalize_player(belief, player)


def _normalize_player(belief, player):
    raw = belief.scores[player]
    total = 0.0
    for role in Role:
        value = float(raw[role])
        if value < 0.0:
            value = 0.0
            raw[role] = 0.0
        total += value

    if total <= 0.0:
        belief.probabilities[player] = {role: 0.0 for role in Role}
        return

    probs = {}
    for role in Role:
        probs[role] = float(raw[role]) / total
    belief.probabilities[player] = probs


def _apply_reveal_constraints(belief, public_state, event, *, strict_no_duplicate_hand):
    revealed_by_player = _collect_revealed_roles(public_state, event)

    for player in sorted(public_state.players):
        alive = int(public_state.players[player].influence_alive)

        if alive <= 0:
            zeroed = {role: 0.0 for role in Role}
            belief.probabilities[player] = zeroed
            belief.scores[player] = dict(zeroed)
            continue

        roles = revealed_by_player.get(player, [])
        if alive == 1 and roles:
            revealed_role = roles[-1]
            probs = dict(belief.probabilities[player])
            if strict_no_duplicate_hand:
                probs[revealed_role] = 0.0
            else:
                probs[revealed_role] *= 0.2

            total = sum(probs.values())
            if total > 0.0:
                for role in Role:
                    probs[role] = probs[role] / total
            else:
                for role in Role:
                    probs[role] = 0.0

            belief.probabilities[player] = probs
            belief.scores[player] = dict(probs)


def _collect_revealed_roles(public_state, event):
    revealed = {player: [] for player in public_state.players}

    for past in public_state.history:
        _append_reveal(revealed, past)
    _append_reveal(revealed, event)
    return revealed


def _append_reveal(revealed, event):
    if isinstance(event, (RevealEvent, InfluenceLostEvent)):
        if event.player in revealed:
            revealed[event.player].append(event.revealed_role)
