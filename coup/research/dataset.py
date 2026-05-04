from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from coup.engine import GameEngine, infer_players
from coup.models import ActionEvent, BlockEvent, ChallengeEvent, parse_events
from coup.rules import action_claim_role, block_claim_roles

ACTION_LABELS = [
    "Income",
    "Foreign Aid",
    "Tax",
    "Steal",
    "Assassinate",
    "Coup",
    "Exchange",
    "NONE",
]


def build_rows_from_events(raw_events, *, game_id, horizon=5):
    events = parse_events(raw_events)
    players = infer_players(events)
    if not players:
        return []

    engine = GameEngine(players)
    rows = []

    for index, event in enumerate(events):
        actor = _actor_for_event(event)
        actor_state = engine.public_state.players.get(actor) if actor else None
        remaining = {role.value: int(value) for role, value in engine.remaining_copies().items()}

        claimant, claimed_role = _primary_claim(event)
        block_type = event.blocked_action if isinstance(event, BlockEvent) else ""
        belief_p_truth = 0.0
        if claimant and claimed_role:
            player_probs = engine.belief_state.probabilities.get(claimant, {})
            for role, prob in player_probs.items():
                if role.value == claimed_role:
                    belief_p_truth = float(prob)
                    break

        row = {
            "meta_game_id": game_id,
            "meta_step": int(index),
            "current_event_type": event.type,
            "current_action_name": getattr(event, "action_name", ""),
            "current_actor": actor or "",
            "current_target": getattr(event, "target", "") or "",
            "current_claimed_role": claimed_role or "",
            "current_block_type": block_type,
            "f_players_alive": int(_players_alive(engine)),
            "f_total_coins": int(sum(player.coins for player in engine.public_state.players.values())),
            "f_actor_coins": int(actor_state.coins) if actor_state else 0,
            "f_actor_influence": int(actor_state.influence_alive) if actor_state else 0,
            "f_remaining_duke": int(remaining.get("Duke", 0)),
            "f_remaining_assassin": int(remaining.get("Assassin", 0)),
            "f_remaining_captain": int(remaining.get("Captain", 0)),
            "f_remaining_ambassador": int(remaining.get("Ambassador", 0)),
            "f_remaining_contessa": int(remaining.get("Contessa", 0)),
            "f_belief_p_truth": float(belief_p_truth),
            "y_next_action": _next_action_label(events, index + 1),
            "y_next_is_challenge": int(
                index + 1 < len(events) and isinstance(events[index + 1], ChallengeEvent)
            ),
            "y_coup_within_horizon": int(_has_coup_within(events, start=index + 1, horizon=horizon)),
            "y_current_claim_challenged": _claim_challenged_label(events, index),
        }

        rows.append(row)
        engine.apply_event(event)

    return rows


def build_dataset_from_sim_dir(sim_dir, *, horizon=5):
    sim_dir = Path(sim_dir)
    rows = []
    for game_path in sorted(sim_dir.glob("game_*.json")):
        raw_events = json.loads(game_path.read_text())
        rows.extend(build_rows_from_events(raw_events, game_id=game_path.stem, horizon=horizon))
    return rows


def write_dataset_csv(rows, out_path):
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


def _next_action_label(events, start):
    for event in events[start:]:
        if isinstance(event, ActionEvent):
            if event.action_name in ACTION_LABELS:
                return event.action_name
            return "NONE"
    return "NONE"


def _has_coup_within(events, *, start, horizon):
    end = min(len(events), start + max(0, int(horizon)))
    for event in events[start:end]:
        if isinstance(event, ActionEvent) and event.action_name == "Coup":
            return True
    return False


def _claim_challenged_label(events, index):
    event = events[index]
    claims = _claims_from_event(event)
    if not claims:
        return -1

    outcomes = _find_claim_outcomes(events, index, claims)
    for value in outcomes.values():
        if value:
            return 1
    return 0


def _primary_claim(event):
    if isinstance(event, ActionEvent):
        role = action_claim_role(event.action_name)
        if role is not None:
            return event.actor, role.value
    elif isinstance(event, BlockEvent):
        roles = block_claim_roles(event.blocked_action)
        if roles:
            return event.blocker, roles[0].value
    return None, None


def _claims_from_event(event):
    claims = []
    if isinstance(event, ActionEvent):
        role = action_claim_role(event.action_name)
        if role is not None:
            claims.append((event.actor, role.value))
    elif isinstance(event, BlockEvent):
        for role in block_claim_roles(event.blocked_action):
            claims.append((event.blocker, role.value))
    return claims


def _find_claim_outcomes(events, claim_index, claims):
    tracked = {claim: False for claim in claims}
    for idx in range(claim_index + 1, len(events)):
        event = events[idx]
        if isinstance(event, (ActionEvent, BlockEvent)):
            break
        if isinstance(event, ChallengeEvent):
            key = (event.challenged, event.claimed_role.value)
            if key in tracked:
                tracked[key] = True
    return tracked


def _players_alive(engine):
    return sum(1 for player in engine.public_state.players.values() if player.influence_alive > 0)


def _actor_for_event(event):
    for attr in ("actor", "blocker", "challenger", "player"):
        value = getattr(event, attr, None)
        if value:
            return value
    return None


def main():
    parser = argparse.ArgumentParser(description="Build research dataset from simulated games")
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    rows = build_dataset_from_sim_dir(args.sim_dir, horizon=args.horizon)
    write_dataset_csv(rows, args.out)
    print(json.dumps({"rows": len(rows), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
