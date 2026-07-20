from __future__ import annotations
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.routes.game import router as game_router, _sessions
from api.ws.manager import ConnectionManager
from coup.models import parse_events

app = FastAPI(
    title="Coup Advisor API",
    description="Real-time multi-player Coup advisor backend",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(game_router)
manager = ConnectionManager()


@app.websocket("/ws/{session_id}/{player_name}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, player_name: str):
    """
    WebSocket endpoint for real-time event streaming.

    Clients connect with their session_id and player_name.
    They can send events as JSON; the server applies them and broadcasts
    the updated state to all connected clients.

    Message format from client:
        {"type": "event", "event": {...coup event dict...}}

    Broadcast from server to all clients:
        {"type": "event_applied", "event": {...}, "by": "player_name",
         "history_length": N}

    Or on error:
        {"type": "error", "detail": "..."}
    """
    await manager.connect(session_id, websocket)
    await manager.broadcast(session_id, {
        "type": "player_joined",
        "player": player_name,
        "connected": manager.session_count(session_id),
    })
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "detail": "Invalid JSON"}))
                continue

            if msg.get("type") == "event":
                engine = _sessions.get(session_id)
                if engine is None:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "detail": f"Session '{session_id}' not found. Create it via POST /game/session/new first.",
                    }))
                    continue
                try:
                    events = parse_events([msg["event"]])
                    engine.apply_event(events[0])
                    await manager.broadcast(session_id, {
                        "type": "event_applied",
                        "event": msg["event"],
                        "by": player_name,
                        "history_length": len(engine.public_state.history),
                    })
                except Exception as exc:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "detail": str(exc),
                    }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "detail": f"Unknown message type: {msg.get('type')}",
                }))

    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)
        await manager.broadcast(session_id, {
            "type": "player_left",
            "player": player_name,
            "connected": manager.session_count(session_id),
        })


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(_sessions)}
