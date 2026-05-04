from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from coup.research.markov import ACTION_SPACE


@dataclass
class StepResult:
    next_state: list[float]
    reward: float
    done: bool
    info: dict


def load_transition_rows(csv_path):
    csv_path = Path(csv_path)
    rows = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = dict(raw)
            for key, value in list(row.items()):
                if value is None or value == "":
                    continue
                if key in {"transition_index", "event_index", "done", "actor_is_winner"}:
                    row[key] = int(float(value))
                elif key == "reward" or key.startswith("s_") or key.startswith("ns_"):
                    row[key] = float(value)
            rows.append(row)
    return rows


class OfflineCoupRLEnv:
    def __init__(self, rows, *, reward_mode="shaped"):
        if reward_mode not in ("shaped", "win"):
            raise ValueError(f"reward_mode must be 'shaped' or 'win', got {reward_mode!r}")
        self.rows = list(rows)
        self.reward_mode = reward_mode
        self.action_labels = list(ACTION_SPACE)

        self.games = {}
        for row in self.rows:
            game_id = row.get("meta_game_id", "game_0")
            self.games.setdefault(game_id, []).append(row)

        for game_rows in self.games.values():
            game_rows.sort(key=lambda item: int(item.get("transition_index", 0)))

        self.game_ids = sorted(self.games.keys())

        if self.rows:
            sample = self.rows[0]
            self.state_keys = sorted(
                key for key in sample.keys() if key.startswith("s_") and not key.startswith("s_ns_")
            )
            self.next_state_keys = sorted(key for key in sample.keys() if key.startswith("ns_"))
        else:
            self.state_keys = []
            self.next_state_keys = []

        self.current_game_id = None
        self.current_rows = []
        self.cursor = 0

    def available_games(self):
        return list(self.game_ids)

    def state_dim(self):
        return len(self.state_keys)

    def action_dim(self):
        return len(self.action_labels)

    def reset(self, game_id=None):
        if not self.games:
            self.current_game_id = None
            self.current_rows = []
            self.cursor = 0
            return [0.0 for _ in self.state_keys]

        if game_id is None or game_id not in self.games:
            game_id = self.game_ids[0]

        self.current_game_id = game_id
        self.current_rows = self.games[game_id]
        self.cursor = 0

        if not self.current_rows:
            return [0.0 for _ in self.state_keys]
        return self._state_vector(self.current_rows[0])

    def step(self, action_index):
        if not self.current_rows:
            return StepResult(
                next_state=[0.0 for _ in self.state_keys],
                reward=0.0,
                done=True,
                info={"logged_action": None, "chosen_action": None},
            )

        row = self.current_rows[self.cursor]
        chosen_action = self.action_labels[int(action_index) % len(self.action_labels)]
        logged_action = row.get("action_name")

        reward = float(row.get("reward", 0.0))
        if self.reward_mode == "shaped":
            if chosen_action == logged_action:
                reward += 1.0
        else:
            if chosen_action == logged_action and int(row.get("actor_is_winner", 0)) == 1:
                reward += 1.0

        done = bool(int(row.get("done", 0)))
        next_state = self._next_state_vector(row)

        info = {
            "game_id": self.current_game_id,
            "transition_index": int(row.get("transition_index", 0)),
            "logged_action": logged_action,
            "chosen_action": chosen_action,
            "action_match": int(chosen_action == logged_action),
        }

        self.cursor += 1
        if self.cursor >= len(self.current_rows):
            done = True

        return StepResult(next_state=next_state, reward=float(reward), done=done, info=info)

    @classmethod
    def from_csv(cls, csv_path, *, reward_mode="shaped"):
        rows = load_transition_rows(csv_path)
        return cls(rows, reward_mode=reward_mode)

    def _state_vector(self, row):
        return [float(row.get(key, 0.0)) for key in self.state_keys]

    def _next_state_vector(self, row):
        if self.next_state_keys:
            return [float(row.get(key, 0.0)) for key in self.next_state_keys]

        fallback = []
        for key in self.state_keys:
            ns_key = key.replace("s_", "ns_", 1)
            fallback.append(float(row.get(ns_key, row.get(key, 0.0))))
        return fallback
