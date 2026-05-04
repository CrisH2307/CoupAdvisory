from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Role(str, Enum):
    DUKE = "Duke"
    ASSASSIN = "Assassin"
    CAPTAIN = "Captain"
    AMBASSADOR = "Ambassador"
    CONTESSA = "Contessa"


class EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str


class ActionEvent(EventBase):
    type: Literal["action"] = "action"
    actor: str
    action_name: str
    target: Optional[str] = None


class BlockEvent(EventBase):
    type: Literal["block"] = "block"
    blocker: str
    blocked_action: str
    roles_claimed: list[Role]


class ChallengeEvent(EventBase):
    type: Literal["challenge"] = "challenge"
    challenger: str
    challenged: str
    claimed_role: Role
    result: Literal["win", "lose"]
    revealed_role: Optional[Role] = None


class RevealEvent(EventBase):
    type: Literal["reveal"] = "reveal"
    player: str
    revealed_role: Role


class CoinChangeEvent(EventBase):
    type: Literal["coin_change"] = "coin_change"
    player: str
    delta: int


class InfluenceLostEvent(EventBase):
    type: Literal["influence_lost"] = "influence_lost"
    player: str
    revealed_role: Role


Event = Annotated[
    Union[
        ActionEvent,
        BlockEvent,
        ChallengeEvent,
        RevealEvent,
        CoinChangeEvent,
        InfluenceLostEvent,
    ],
    Field(discriminator="type"),
]

_EVENT_LIST_ADAPTER = TypeAdapter(list[Event])


class PlayerPublicState(BaseModel):
    coins: int = 2
    influence_alive: int = 2


class PublicState(BaseModel):
    players: dict[str, PlayerPublicState]
    revealed_dead: dict[Role, int]
    history: list[Event] = Field(default_factory=list)

    @classmethod
    def fresh(cls, player_names):
        players = {name: PlayerPublicState() for name in player_names}
        revealed_dead = {role: 0 for role in Role}
        return cls(players=players, revealed_dead=revealed_dead, history=[])


class BeliefState(BaseModel):
    scores: dict[str, dict[Role, float]]
    probabilities: dict[str, dict[Role, float]]

    @classmethod
    def fresh(cls, player_names):
        scores = {}
        probabilities = {}
        for name in player_names:
            scores[name] = {role: 1.0 for role in Role}
            probabilities[name] = {role: 1.0 / len(Role) for role in Role}
        return cls(scores=scores, probabilities=probabilities)


def parse_events(raw_events):
    return _EVENT_LIST_ADAPTER.validate_python(raw_events)
