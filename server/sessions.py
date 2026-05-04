from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict

from coup.engine import GameEngine
from coup.models import Role


@dataclass
class GameSession:
    session_id: str
    engine: GameEngine
    player_names: list[str]
    strict_no_duplicate_hand: bool = False
    self_seat: int | None = None
    self_hand: list[Role] = field(default_factory=list)
    subscribers: list = field(default_factory=list)

    def self_player(self) -> str | None:
        if self.self_seat is None:
            return None
        if 0 <= self.self_seat < len(self.player_names):
            return self.player_names[self.self_seat]
        return None

    def apply_self_pin_to_engine(self) -> None:
        player = self.self_player()
        if player is None:
            self.engine.set_self_pin(None, [])
        else:
            self.engine.set_self_pin(player, self.self_hand)


class SessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, GameSession] = {}

    def create(self, player_names, strict_no_duplicate_hand=False) -> GameSession:
        sid = uuid.uuid4().hex[:12]
        engine = GameEngine(player_names, strict_no_duplicate_hand=strict_no_duplicate_hand)
        session = GameSession(
            session_id=sid,
            engine=engine,
            player_names=list(player_names),
            strict_no_duplicate_hand=strict_no_duplicate_hand,
        )
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, sid: str) -> GameSession | None:
        with self._lock:
            return self._sessions.get(sid)

    def require(self, sid: str) -> GameSession:
        session = self.get(sid)
        if session is None:
            raise KeyError(f"session not found: {sid}")
        return session

    def drop(self, sid: str) -> bool:
        with self._lock:
            return self._sessions.pop(sid, None) is not None

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())


store = SessionStore()
