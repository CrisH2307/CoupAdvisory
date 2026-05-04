from __future__ import annotations

import threading
import uuid

from fastapi import APIRouter, HTTPException

from coup.models import parse_events
from coup.sim.bots import BotStyle
from coup.sim.sim import simulate_games
from coup.sim.metrics import summarize_game

from ..schemas import SimGameResult, SimRunRequest, SimRunResponse

router = APIRouter(prefix="/api/sim", tags=["sim"])

_runs: dict[str, SimRunResponse] = {}
_lock = threading.Lock()


@router.get("/bots")
def list_bot_styles() -> dict:
    return {"styles": [s.value for s in BotStyle]}


@router.post("/run", response_model=SimRunResponse)
def run_sim(body: SimRunRequest) -> SimRunResponse:
    try:
        results = simulate_games(
            body.games,
            players=body.players,
            seed=body.seed,
            bot_mix=body.bot_mix,
            max_turns=body.max_turns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    run_id = uuid.uuid4().hex[:12]
    games = []
    summary_rows = []
    for index, result in enumerate(results, start=1):
        parsed = parse_events(result.events)
        games.append(
            SimGameResult(
                game_index=index,
                winner=result.winner,
                turns=result.turns,
                events=parsed,
            )
        )
        metrics = summarize_game(
            events=result.events,
            players=result.players,
            winner=result.winner,
            turns=result.turns,
            advisor_policy=body.advisor_policy,
        )
        from dataclasses import asdict as _asdict

        row = {"game": index}
        row.update(_asdict(metrics))
        row.pop("reveal_probability_bins", None)
        summary_rows.append(row)

    response = SimRunResponse(run_id=run_id, games=games, summary_rows=summary_rows)
    with _lock:
        _runs[run_id] = response
    return response


@router.get("/{run_id}", response_model=SimRunResponse)
def get_run(run_id: str) -> SimRunResponse:
    with _lock:
        run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@router.get("/{run_id}/games/{game_index}", response_model=SimGameResult)
def get_run_game(run_id: str, game_index: int) -> SimGameResult:
    with _lock:
        run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    for game in run.games:
        if game.game_index == game_index:
            return game
    raise HTTPException(status_code=404, detail="game index not in run")
