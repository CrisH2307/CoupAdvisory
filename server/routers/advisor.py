from __future__ import annotations

from fastapi import APIRouter, HTTPException

from coup.advisor import (
    ClaimContext,
    Recommendation,
    _threshold_for_claim,
    recommend_challenge,
)
from coup.advisor_ml import recommend_challenge_ml

from ..schemas import AdvisorRequest, AdvisorResponse
from ..sessions import store

router = APIRouter(prefix="/api/advisor", tags=["advisor"])


@router.post("/recommend", response_model=AdvisorResponse)
def recommend(body: AdvisorRequest) -> AdvisorResponse:
    if body.session_id is None:
        raise HTTPException(status_code=400, detail="session_id required for recommend endpoint")
    session = store.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    remaining = session.engine.remaining_copies().get(body.claimed_role, 0)
    claim_context = ClaimContext(
        action_name=body.action_name,
        block_type=body.block_type,
        remaining_copies=int(remaining),
    )

    try:
        if body.engine == "heuristic":
            result = recommend_challenge(
                session.engine.belief_state,
                body.claimant,
                body.claimed_role,
                claim_context,
                style=body.style,
                public_state=session.engine.public_state,
            )
        elif body.engine == "ml":
            result = recommend_challenge_ml(
                session.engine.belief_state,
                body.claimant,
                body.claimed_role,
                claim_context,
                style=body.style,
                public_state=session.engine.public_state,
            )
        elif body.engine == "ensemble":
            heuristic_result = recommend_challenge(
                session.engine.belief_state,
                body.claimant,
                body.claimed_role,
                claim_context,
                style=body.style,
                public_state=session.engine.public_state,
            )
            ml_result = recommend_challenge_ml(
                session.engine.belief_state,
                body.claimant,
                body.claimed_role,
                claim_context,
                style=body.style,
                public_state=session.engine.public_state,
            )
            result = _ensemble(heuristic_result, ml_result, claim_context, body.style)
        else:
            raise HTTPException(status_code=400, detail=f"unknown engine: {body.engine}")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="ml_model_not_trained") from exc

    return AdvisorResponse(
        recommendation=result.recommendation,
        explanation=result.explanation,
        p_truth=result.p_truth,
        threshold=result.threshold,
        engine=body.engine,
    )


def _ensemble(
    heuristic_result: Recommendation,
    ml_result: Recommendation,
    claim_context: ClaimContext,
    style: str,
) -> Recommendation:
    p_truth = (heuristic_result.p_truth + ml_result.p_truth) / 2.0
    threshold = _threshold_for_claim(claim_context, style=style)
    recommendation = "Challenge" if p_truth < threshold else "Do not challenge"
    explanation = (
        f"engine=ensemble, p_truth={p_truth:.2f} = mean("
        f"heuristic={heuristic_result.p_truth:.2f}, "
        f"ml={ml_result.p_truth:.2f}) vs threshold={threshold:.2f}"
    )
    return Recommendation(
        recommendation=recommendation,
        explanation=explanation,
        p_truth=p_truth,
        threshold=threshold,
    )
