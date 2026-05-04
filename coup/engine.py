from __future__ import annotations

from coup.belief import apply_event as apply_belief_event
from coup.models import (
    ActionEvent,
    BeliefState,
    CoinChangeEvent,
    InfluenceLostEvent,
    PublicState,
    RevealEvent,
    Role,
)


def infer_players(events):
    players = set()
    for event in events:
        if isinstance(event, ActionEvent):
            players.add(event.actor)
            if event.target:
                players.add(event.target)
        if isinstance(event, CoinChangeEvent):
            players.add(event.player)
        if isinstance(event, InfluenceLostEvent):
            players.add(event.player)
        if isinstance(event, RevealEvent):
            players.add(event.player)

        for attr in ("blocker", "challenger", "challenged"):
            value = getattr(event, attr, None)
            if value:
                players.add(value)
    return sorted(players)


class GameEngine:
    def __init__(self, player_names, *, strict_no_duplicate_hand=False):
        self.public_state = PublicState.fresh(player_names)
        self.belief_state = BeliefState.fresh(player_names)
        self.strict_no_duplicate_hand = strict_no_duplicate_hand
        self.self_pin = None

    def set_self_pin(self, player, hand):
        if player is None:
            self.self_pin = None
        else:
            self.self_pin = (player, list(hand))

    def apply_event(self, event):
        self._apply_public(event)
        apply_belief_event(
            self.belief_state,
            self.public_state,
            event,
            strict_no_duplicate_hand=self.strict_no_duplicate_hand,
            self_pin=self.self_pin,
        )
        self.public_state.history.append(event)

    def replay(self, events):
        for event in events:
            self.apply_event(event)

    def remaining_copies(self):
        remaining = {}
        for role in Role:
            remaining[role] = max(0, 3 - int(self.public_state.revealed_dead.get(role, 0)))
        return remaining

    def _apply_public(self, event):
        if isinstance(event, CoinChangeEvent):
            player_state = self.public_state.players.get(event.player)
            if player_state is None:
                return
            player_state.coins = max(0, int(player_state.coins) + int(event.delta))
            return

        if isinstance(event, (InfluenceLostEvent, RevealEvent)):
            self._record_reveal(event.player, event.revealed_role)

    def _record_reveal(self, player, role):
        player_state = self.public_state.players.get(player)
        if player_state is None:
            return
        player_state.influence_alive = max(0, int(player_state.influence_alive) - 1)
        self.public_state.revealed_dead[role] = int(self.public_state.revealed_dead.get(role, 0)) + 1
