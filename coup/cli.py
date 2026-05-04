from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabulate import tabulate

from coup.advisor import ClaimContext, recommend_challenge
from coup.engine import GameEngine, infer_players
from coup.models import ActionEvent, BlockEvent, InfluenceLostEvent, RevealEvent, Role, parse_events
from coup.rules import action_claim_role, block_claim_roles
from vision.state_adapter import build_diff_events, pipeline_to_public_state


def main():
    parser = argparse.ArgumentParser(description="Coup observer + advisor")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--events", help="Path to JSON list of events")
    source_group.add_argument("--vision-pipeline", help="Path to vision pipeline_output.json")
    parser.add_argument(
        "--previous-state",
        help="Optional previous state_from_vision.json for diff events",
    )
    parser.add_argument(
        "--strict-no-duplicate-hand",
        action="store_true",
        help="Enforce that a player's hidden card cannot match their revealed role.",
    )
    parser.add_argument(
        "--advisor-style",
        choices=["Conservative", "Balanced", "Aggressive"],
        default="Balanced",
        help="Challenge style for advisor thresholds.",
    )
    args = parser.parse_args()

    if args.events:
        _run_events_mode(
            events_path=Path(args.events),
            strict_no_duplicate_hand=args.strict_no_duplicate_hand,
            advisor_style=args.advisor_style,
        )
        return

    _run_vision_mode(
        pipeline_path=Path(args.vision_pipeline),
        previous_state_path=Path(args.previous_state) if args.previous_state else None,
        strict_no_duplicate_hand=args.strict_no_duplicate_hand,
        advisor_style=args.advisor_style,
    )


def _run_events_mode(events_path, *, strict_no_duplicate_hand, advisor_style="Balanced"):
    raw_events = json.loads(events_path.read_text())
    events = parse_events(raw_events)
    players = infer_players(events)
    engine = GameEngine(players, strict_no_duplicate_hand=strict_no_duplicate_hand)
    engine.replay(events)

    _print_public_state(engine.public_state)
    _print_revealed_roles(engine.public_state)
    _print_belief_mode(strict_no_duplicate_hand)
    _print_hidden_probabilities(engine)
    _print_advisor(engine, events[-1] if events else None, advisor_style=advisor_style)


def _run_vision_mode(
    pipeline_path,
    previous_state_path,
    strict_no_duplicate_hand,
    advisor_style="Balanced",
):
    pipeline_payload = json.loads(Path(pipeline_path).read_text())
    public_state, generated_events, revealed_roles = pipeline_to_public_state(pipeline_payload)

    if previous_state_path is not None:
        previous_payload = json.loads(Path(previous_state_path).read_text())
        raw_events = build_diff_events(previous_payload, public_state, revealed_roles)
    else:
        raw_events = generated_events

    events = parse_events(raw_events)

    players = list(public_state.players.keys())
    if not players:
        players = infer_players(events)

    engine = GameEngine(players, strict_no_duplicate_hand=strict_no_duplicate_hand)
    engine.replay(events)

    _print_public_state(engine.public_state)
    _print_revealed_roles(engine.public_state, revealed_override=revealed_roles)
    _print_belief_mode(strict_no_duplicate_hand)
    _print_hidden_probabilities(engine)
    _print_advisor(engine, events[-1] if events else None, advisor_style=advisor_style)


def _print_public_state(public_state):
    print("\nPublic State")
    rows = []
    for player in sorted(public_state.players, key=_player_sort_key):
        state = public_state.players[player]
        rows.append([player, state.coins, state.influence_alive])
    print(tabulate(rows, headers=["Player", "Coins", "Influence"], tablefmt="github"))

    print("\nRevealed Dead")
    role_rows = []
    for role in Role:
        revealed = int(public_state.revealed_dead.get(role, 0))
        remaining = max(0, 3 - revealed)
        role_rows.append([role.value, revealed, remaining])
    print(tabulate(role_rows, headers=["Role", "Revealed", "Remaining"], tablefmt="github"))


def _print_revealed_roles(public_state, *, revealed_override=None):
    print("\nRevealed Roles")
    revealed_map = _collect_revealed_map(public_state)
    if revealed_override is not None:
        for player, roles in revealed_override.items():
            revealed_map[player] = list(roles)

    rows = []
    for player in sorted(public_state.players, key=_player_sort_key):
        state = public_state.players[player]
        roles = revealed_map.get(player, [])
        shown = ", ".join(roles) if roles else "-"

        if state.influence_alive <= 0:
            status = "ELIMINATED"
        else:
            status = f"{state.influence_alive} hidden left"

        rows.append([player, shown, status])

    print(tabulate(rows, headers=["Player", "Revealed", "Status"], tablefmt="github"))


def _print_belief_mode(strict_no_duplicate_hand):
    if strict_no_duplicate_hand:
        mode = "Strict no-duplicate-hand"
    else:
        mode = "Standard Coup (duplicates allowed)"
    print(f"\nBelief mode: {mode}")


def _print_hidden_probabilities(engine):
    print("\nHidden Role Probabilities")
    headers = ["Player"] + [role.value for role in Role]
    rows = []

    for player in sorted(engine.public_state.players, key=_player_sort_key):
        state = engine.public_state.players[player]
        if state.influence_alive <= 0:
            rows.append([player] + ["ELIM"] * len(Role))
            continue

        probs = engine.belief_state.probabilities.get(player, {})
        row = [player]
        for role in Role:
            row.append(f"{float(probs.get(role, 0.0)):.2f}")
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="github"))


def _print_advisor(engine, latest_event, *, advisor_style="Balanced"):
    claims = _claims_from_event(latest_event)
    if not claims:
        return

    print("\nAdvisor Claim Review")
    rows = []
    remaining = engine.remaining_copies()

    for claim in claims:
        context = ClaimContext(
            action_name=claim["action_name"],
            block_type=claim["block_type"],
            remaining_copies=remaining.get(claim["role"], 0),
        )
        rec = recommend_challenge(
            engine.belief_state,
            claim["claimant"],
            claim["role"],
            context,
            style=advisor_style,
        )
        rows.append(
            [
                claim["claimant"],
                claim["role"].value,
                rec.recommendation,
                f"{rec.p_truth:.2f}",
                f"{rec.threshold:.2f}",
                rec.explanation,
            ]
        )

    print(
        tabulate(
            rows,
            headers=["Player", "Claimed Role", "Advice", "p_truth", "threshold", "note"],
            tablefmt="github",
        )
    )


def _claims_from_event(event):
    claims = []
    if isinstance(event, ActionEvent):
        role = action_claim_role(event.action_name)
        if role is not None:
            claims.append(
                {
                    "claimant": event.actor,
                    "role": role,
                    "action_name": event.action_name,
                    "block_type": None,
                }
            )
    elif isinstance(event, BlockEvent):
        for role in block_claim_roles(event.blocked_action):
            claims.append(
                {
                    "claimant": event.blocker,
                    "role": role,
                    "action_name": None,
                    "block_type": event.blocked_action,
                }
            )
    return claims


def _collect_revealed_map(public_state):
    revealed_map = {player: [] for player in public_state.players}
    for event in public_state.history:
        if isinstance(event, (InfluenceLostEvent, RevealEvent)):
            if event.player in revealed_map:
                revealed_map[event.player].append(event.revealed_role.value)
    return revealed_map


def _player_sort_key(name):
    if isinstance(name, str) and name.startswith("P") and name[1:].isdigit():
        return (0, int(name[1:]))
    return (1, str(name))


if __name__ == "__main__":
    main()
