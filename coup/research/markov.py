from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from coup.engine import GameEngine, infer_players
from coup.models import ActionEvent, parse_events


ACTION_SPACE = ["Income", "Foreign Aid", "Tax", "Steal", "Assassinate", "Coup", "Exchange"]


def encode_markov_state(engine, *, actor=None, ordered_players=None):
    players = ordered_players or sorted(engine.public_state.players.keys())

    state = {
        "players_alive": float(
            sum(1 for player in engine.public_state.players.values() if player.influence_alive > 0)
        ),
        "total_coins": float(sum(player.coins for player in engine.public_state.players.values())),
        "actor_coins": 0.0,
        "actor_influence": 0.0,
    }

    if actor in engine.public_state.players:
        actor_state = engine.public_state.players[actor]
        state["actor_coins"] = float(actor_state.coins)
        state["actor_influence"] = float(actor_state.influence_alive)

    for name in players:
        player = engine.public_state.players[name]
        key = name.lower()
        state[f"{key}_coins"] = float(player.coins)
        state[f"{key}_influence"] = float(player.influence_alive)

    for role, remaining in engine.remaining_copies().items():
        state[f"remaining_{role.value.lower()}"] = float(remaining)

    return state


def build_action_transitions_from_events(raw_events, *, game_id, reward_mode="shaped"):
    if reward_mode not in ("shaped", "win"):
        raise ValueError(f"reward_mode must be 'shaped' or 'win', got {reward_mode!r}")

    events = parse_events(raw_events)
    players = infer_players(events)
    if not players:
        return []

    engine = GameEngine(players)
    rows = []
    transition_index = 0

    index = 0
    while index < len(events):
        event = events[index]

        if not isinstance(event, ActionEvent):
            engine.apply_event(event)
            index += 1
            continue

        state = encode_markov_state(engine, actor=event.actor, ordered_players=players)
        pre_coins = state["actor_coins"]
        pre_alive = state["players_alive"]

        engine.apply_event(event)
        next_index = index + 1
        while next_index < len(events) and not isinstance(events[next_index], ActionEvent):
            engine.apply_event(events[next_index])
            next_index += 1

        next_state = encode_markov_state(engine, actor=event.actor, ordered_players=players)
        shaped_reward = (next_state["actor_coins"] - pre_coins) + (pre_alive - next_state["players_alive"])
        done = int(next_state["players_alive"] <= 1 or next_index >= len(events))

        reward = float(shaped_reward) if reward_mode == "shaped" else 0.0

        row = {
            "meta_game_id": game_id,
            "transition_index": int(transition_index),
            "event_index": int(index),
            "actor": event.actor,
            "action_name": event.action_name,
            "reward": float(reward),
            "done": int(done),
            "s_players_alive": float(state["players_alive"]),
            "ns_players_alive": float(next_state["players_alive"]),
        }

        for key, value in state.items():
            row[f"s_{key}"] = float(value)
        for key, value in next_state.items():
            row[f"ns_{key}"] = float(value)

        rows.append(row)
        transition_index += 1
        index = next_index

    alive_players = [
        name
        for name, state in engine.public_state.players.items()
        if state.influence_alive > 0
    ]
    winner = alive_players[0] if len(alive_players) == 1 else None
    for row in rows:
        row["actor_is_winner"] = int(winner is not None and row["actor"] == winner)

    if reward_mode == "win":
        for row in rows:
            row["reward"] = 0.0

    return rows


def build_transition_dataset_from_sim_dir(sim_dir, *, reward_mode="shaped"):
    sim_dir = Path(sim_dir)
    rows = []
    for game_path in sorted(sim_dir.glob("game_*.json")):
        raw_events = json.loads(game_path.read_text())
        rows.extend(
            build_action_transitions_from_events(
                raw_events, game_id=game_path.stem, reward_mode=reward_mode
            )
        )
    return rows


def write_transition_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        out_path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build Markov transition dataset")
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--reward-mode",
        choices=["shaped", "win"],
        default="shaped",
        help="shaped = legacy per-step (Δcoins + Δalive); win = sparse terminal win reward.",
    )
    args = parser.parse_args()

    rows = build_transition_dataset_from_sim_dir(args.sim_dir, reward_mode=args.reward_mode)
    write_transition_csv(rows, args.out)
    print(
        json.dumps(
            {"rows": len(rows), "out": str(args.out), "reward_mode": args.reward_mode},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
