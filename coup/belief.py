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
    _apply_global_count_constraint(belief, public_state)
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
    entry = ACTION_ROLE_MULTIPLIERS.get(event.action_name)
    if entry is None:
        return
    role, factor = entry
    belief.scores[event.actor][role] *= float(factor)


def _apply_block_multiplier(belief, event):
    if event.blocker not in belief.scores:
        return
    for role, factor in BLOCK_MULTIPLIERS.get(event.blocked_action, []):
        belief.scores[event.blocker][role] *= float(factor)


def _apply_challenge_multiplier(belief, event):
    if event.challenged not in belief.scores:
        return
    if event.result == "win":
        scale = 10.0
    else:
        scale = 0.01
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
def _apply_global_count_constraint(belief, public_state):
    """
    Rescale each role's probability across all alive players so that the
    expected number of that role in all hands equals the remaining deck count.

    This enforces the finite-deck constraint that individual normalization
    misses: sum_over_players(p[player][role] * hand_size[player]) <= remaining[role].
    """
    alive_players = [
        name for name, state in public_state.players.items()
        if int(state.influence_alive) > 0
    ]
    if not alive_players:
        return

    for role in Role:
        revealed = int(public_state.revealed_dead.get(role, 0))
        remaining = max(0, 3 - revealed)

        # Compute expected count of this role across all alive hands
        expected = 0.0
        for name in alive_players:
            hand_size = int(public_state.players[name].influence_alive)
            p = float(belief.probabilities.get(name, {}).get(role, 0.0))
            expected += p * hand_size

        if expected <= 0.0:
            continue

        # Scale so expected count matches remaining copies
        scale = remaining / expected
        # Cap scale: don't inflate beyond 2.0 or deflate below 0.0
        scale = max(0.0, min(scale, 2.0))
        if abs(scale - 1.0) < 1e-6:
            continue

        for name in alive_players:
            if name not in belief.probabilities:
                continue
            old_p = float(belief.probabilities[name].get(role, 0.0))
            belief.probabilities[name][role] = old_p * scale

        # Re-normalize each player so their probabilities sum to 1
        for name in alive_players:
            _normalize_player(belief, name)


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
