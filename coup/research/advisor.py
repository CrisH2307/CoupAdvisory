from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict
from coup.advisor import recommend_challenge as _core_recommend_challenge


class ClaimContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_name: Optional[str] = None
    block_type: Optional[str] = None
    remaining_copies: int


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recommendation: Literal["Challenge", "Do not challenge"]
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
):
    core = _core_recommend_challenge(
        belief_state,
        claimant,
        claimed_role,
        claim_context,
        style=style,
    )
    return Recommendation(
        recommendation=core.recommendation,
        explanation=core.explanation,
        p_truth=float(core.p_truth),
        threshold=float(core.threshold),
    )
