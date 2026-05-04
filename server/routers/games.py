from __future__ import annotations

from fastapi import APIRouter, HTTPException

from coup.engine import GameEngine
from coup.models import Event, parse_events

from ..schemas import (
    CreateGameRequest,
    GameStateResponse,
    ReplayRequest,
    SelfHandRequest,
    SelfSeatRequest,
)
from ..sessions import store

router = APIRouter(prefix="/api/games", tags=["games"])


def _snapshot(session) -> GameStateResponse:
    engine = session.engine
    return GameStateResponse(
        session_id=session.session_id,
        player_names=session.player_names,
        public_state=engine.public_state,
        belief_state=engine.belief_state,
        remaining_copies=engine.remaining_copies(),
        strict_no_duplicate_hand=session.strict_no_duplicate_hand,
        self_seat=session.self_seat,
        self_hand=list(session.self_hand),
    )


@router.post("", response_model=GameStateResponse)
def create_game(body: CreateGameRequest) -> GameStateResponse:
    session = store.create(body.player_names, body.strict_no_duplicate_hand)
    return _snapshot(session)


@router.get("/{sid}", response_model=GameStateResponse)
def get_game(sid: str) -> GameStateResponse:
    session = store.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return _snapshot(session)


@router.post("/{sid}/events", response_model=GameStateResponse)
def apply_event(sid: str, event: Event) -> GameStateResponse:
    session = store.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    session.engine.apply_event(event)
    return _snapshot(session)


@router.post("/{sid}/replay", response_model=GameStateResponse)
def replay_events(sid: str, body: ReplayRequest) -> GameStateResponse:
    session = store.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    session.engine = GameEngine(
        session.player_names,
        strict_no_duplicate_hand=session.strict_no_duplicate_hand,
    )
    session.apply_self_pin_to_engine()
    session.engine.replay(body.events)
    return _snapshot(session)


@router.post("/{sid}/self", response_model=GameStateResponse)
def set_self_seat(sid: str, body: SelfSeatRequest) -> GameStateResponse:
    session = store.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    if len(session.engine.public_state.history) > 0:
        raise HTTPException(
            status_code=400,
            detail="cannot set self seat after events have been applied; reset the game first",
        )
    if body.seat is not None and not (0 <= body.seat < len(session.player_names)):
        raise HTTPException(status_code=400, detail="seat out of range")
    session.self_seat = body.seat
    session.self_hand = list(body.hand)
    session.apply_self_pin_to_engine()
    session.engine.belief_state = type(session.engine.belief_state).fresh(session.player_names)
    if session.self_seat is not None and session.self_hand:
        from coup.belief import _apply_self_pin

        _apply_self_pin(
            session.engine.belief_state,
            session.engine.public_state,
            (session.player_names[session.self_seat], session.self_hand),
        )
    return _snapshot(session)


@router.patch("/{sid}/self/hand", response_model=GameStateResponse)
def update_self_hand(sid: str, body: SelfHandRequest) -> GameStateResponse:
    session = store.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    if session.self_seat is None:
        raise HTTPException(status_code=400, detail="self seat is not set")
    session.self_hand = list(body.hand)
    session.apply_self_pin_to_engine()
    from coup.belief import _apply_self_pin

    _apply_self_pin(
        session.engine.belief_state,
        session.engine.public_state,
        (session.player_names[session.self_seat], session.self_hand),
    )
    return _snapshot(session)


@router.delete("/{sid}")
def delete_game(sid: str) -> dict:
    if not store.drop(sid):
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True}


@router.post("/{sid}/reset", response_model=GameStateResponse)
def reset_game(sid: str) -> GameStateResponse:
    session = store.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    session.engine = GameEngine(
        session.player_names,
        strict_no_duplicate_hand=session.strict_no_duplicate_hand,
    )
    session.apply_self_pin_to_engine()
    if session.self_seat is not None and session.self_hand:
        from coup.belief import _apply_self_pin

        _apply_self_pin(
            session.engine.belief_state,
            session.engine.public_state,
            (session.player_names[session.self_seat], session.self_hand),
        )
    return _snapshot(session)
