from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from coup.engine import GameEngine
from coup.models import parse_events

try:
    from coup.advisor import advisor_brief as _advisor_brief
except ImportError:
    _advisor_brief = None

router = APIRouter(prefix="/game", tags=["game"])

# In-memory game sessions: session_id → GameEngine
# In production, replace with a proper store (Redis, DB, etc.)
_sessions: dict[str, GameEngine] = {}


class NewSessionRequest(BaseModel):
    session_id: str
    player_names: list[str]
    strict_no_duplicate_hand: bool = False


class EventRequest(BaseModel):
    session_id: str
    event: dict


class AdvisorRequest(BaseModel):
    session_id: str
    perspective: str
    style: str = "Balanced"


@router.post("/session/new")
def create_session(req: NewSessionRequest):
    if req.session_id in _sessions:
        raise HTTPException(status_code=409, detail="Session already exists")
    engine = GameEngine(
        req.player_names,
        strict_no_duplicate_hand=req.strict_no_duplicate_hand,
    )
    _sessions[req.session_id] = engine
    return {"session_id": req.session_id, "players": req.player_names}


@router.get("/session/{session_id}/state")
def get_state(session_id: str):
    engine = _sessions.get(session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "players": {
            name: {
                "coins": int(state.coins),
                "influence_alive": int(state.influence_alive),
            }
            for name, state in engine.public_state.players.items()
        },
        "revealed_dead": {
            role.value: int(count)
            for role, count in engine.public_state.revealed_dead.items()
        },
        "history_length": len(engine.public_state.history),
    }


@router.post("/session/{session_id}/event")
def apply_event(session_id: str, req: EventRequest):
    engine = _sessions.get(session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        events = parse_events([req.event])
        engine.apply_event(events[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"ok": True, "history_length": len(engine.public_state.history)}


@router.post("/session/{session_id}/advisor")
def get_advisor_brief(session_id: str, req: AdvisorRequest):
    engine = _sessions.get(session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if _advisor_brief is None:
        return {
            "warnings": [
                "advisor_brief() is not available in this build; returning minimal session summary."
            ],
            "perspective": req.perspective,
            "style": req.style,
            "history_length": len(engine.public_state.history),
        }
    brief = _advisor_brief(
        engine.public_state,
        engine.belief_state,
        perspective=req.perspective,
        style=req.style,
    )
    return brief.model_dump()


@router.get("/sessions")
def list_sessions():
    return {"sessions": list(_sessions.keys())}
