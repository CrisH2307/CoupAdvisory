from __future__ import annotations

from coup.belief import apply_event as apply_belief_event
from coup.models import (
    ActionEvent,
    BeliefState,
    CoinChangeEvent,
    InfluenceLostEvent,
    PublicState,
    RevealEvent,
)


def infer_players(events):
    players: set[str] = set()
    for event in events:
        if isinstance(event, ActionEvent):
            players.add(event.actor)
            if event.target:
                players.add(event.target)
        elif isinstance(event, CoinChangeEvent):
            players.add(event.player)
        elif isinstance(event, InfluenceLostEvent):
            players.add(event.player)
        elif isinstance(event, RevealEvent):
            players.add(event.player)
        else:
            for attr in ("blocker", "challenger", "challenged"):
                name = getattr(event, attr, None)
                if name:
                    players.add(name)
    return sorted(players)


class GameEngine:
    def __init__(self, player_names, *, strict_no_duplicate_hand=False):
        self.public_state = PublicState.fresh(player_names)
        self.belief_state = BeliefState.fresh(player_names)
        self.strict_no_duplicate_hand = strict_no_duplicate_hand

    def apply_event(self, event):
        self._apply_public(event)
        apply_belief_event(
            self.belief_state,
            self.public_state,
            event,
            strict_no_duplicate_hand=self.strict_no_duplicate_hand,
        )
        self.public_state.history.append(event)

    def replay(self, events):
        for event in events:
            self.apply_event(event)

    def remaining_copies(self):
        return {
            role: max(0, 3 - self.public_state.revealed_dead.get(role, 0))
            for role in self.public_state.revealed_dead
        }

    def _apply_public(self, event):
        if isinstance(event, CoinChangeEvent):
            self.public_state.players[event.player].coins += event.delta
        elif isinstance(event, InfluenceLostEvent):
            self._record_reveal(event.player, event.revealed_role)
        elif isinstance(event, RevealEvent):
            self._record_reveal(event.player, event.revealed_role)

    def _record_reveal(self, player, role):
        self.public_state.players[player].influence_alive = max(
            0, self.public_state.players[player].influence_alive - 1
        )
        self.public_state.revealed_dead[role] += 1
