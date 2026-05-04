from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from coup.models import BeliefState, Event, PublicState, Role


class CreateGameRequest(BaseModel):
    player_names: list[str] = Field(..., min_length=2, max_length=6)
    strict_no_duplicate_hand: bool = False


class GameStateResponse(BaseModel):
    session_id: str
    player_names: list[str]
    public_state: PublicState
    belief_state: BeliefState
    remaining_copies: dict[Role, int]
    strict_no_duplicate_hand: bool
    self_seat: Optional[int] = None
    self_hand: list[Role] = Field(default_factory=list)


class SelfSeatRequest(BaseModel):
    seat: Optional[int] = None
    hand: list[Role] = Field(default_factory=list, max_length=2)


class SelfHandRequest(BaseModel):
    hand: list[Role] = Field(default_factory=list, max_length=2)


class ReplayRequest(BaseModel):
    events: list[Event]


class AdvisorRequest(BaseModel):
    session_id: Optional[str] = None
    claimant: str
    claimed_role: Role
    action_name: Optional[str] = None
    block_type: Optional[str] = None
    style: Literal["Conservative", "Balanced", "Aggressive"] = "Balanced"
    engine: Literal["heuristic", "ml", "ensemble"] = "heuristic"


class AdvisorResponse(BaseModel):
    recommendation: str
    explanation: str
    p_truth: float
    threshold: float
    engine: str


class SimRunRequest(BaseModel):
    players: int = Field(4, ge=2, le=6)
    games: int = Field(20, ge=1, le=500)
    seed: int = 0
    max_turns: int = 200
    bot_mix: Optional[list[str]] = None
    advisor_policy: Literal["advisor", "never_challenge", "always_challenge"] = "advisor"


class SimGameResult(BaseModel):
    game_index: int
    winner: Optional[str]
    turns: int
    events: list[Event]


class SimRunResponse(BaseModel):
    run_id: str
    games: list[SimGameResult]
    summary_rows: list[dict]


class MetricsSnapshot(BaseModel):
    policy: str
    games: int
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    accuracy: Optional[float] = None
    outcome_accuracy: Optional[float] = None
    extra: dict = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    policies: list[MetricsSnapshot]
