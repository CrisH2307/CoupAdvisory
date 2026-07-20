from __future__ import annotations
import json
from fastapi import WebSocket


class ConnectionManager:
    """
    Manages WebSocket connections per game session.
    Broadcasts events to all clients in the same session.
    """

    def __init__(self):
        # session_id → list of active WebSocket connections
        self._sessions: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._sessions.setdefault(session_id, []).append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        connections = self._sessions.get(session_id, [])
        if websocket in connections:
            connections.remove(websocket)
        if not connections:
            self._sessions.pop(session_id, None)

    async def broadcast(self, session_id: str, message: dict) -> None:
        """Send a JSON message to all clients in a session."""
        connections = list(self._sessions.get(session_id, []))
        dead = []
        for ws in connections:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(session_id, ws)

    def session_count(self, session_id: str) -> int:
        return len(self._sessions.get(session_id, []))

    def all_sessions(self) -> list[str]:
        return list(self._sessions.keys())
