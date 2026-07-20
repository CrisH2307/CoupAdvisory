from __future__ import annotations

"""
Outcome-based labeling for challenge decisions in Coup game logs.

For each opportunity to challenge (action claim or block claim), determines:
  - Was a challenge made?
  - What was the influence differential for the challenger within N turns?
  - Label: 1 if challenging led to net positive outcome, 0 if not challenging
           (or challenging led to net negative) was better.

Usage:
    from coup.research.outcome_labels import label_challenge_opportunities
    rows = label_challenge_opportunities(events, winner)
"""

from coup.models import ActionEvent, BlockEvent, ChallengeEvent, parse_events
from coup.engine import GameEngine, infer_players
from coup.rules import ACTION_ROLE_MULTIPLIERS, BLOCK_MULTIPLIERS


OUTCOME_WINDOW = 5  # look N events forward to assess challenge outcome


def label_challenge_opportunities(raw_events: list[dict], winner: str | None) -> list[dict]:
    """
    Scan a game's event log and produce one row per challenge opportunity.
    Each row contains:
      - Features: belief state, remaining_copies, actor, role, context
      - label_optimal_challenge: 1 if the optimal decision was to challenge
      - label_was_challenged: 1 if a challenge actually happened
      - label_outcome_positive: 1 if the actual decision led to net positive result

    Args:
        raw_events: list of raw event dicts from a game
        winner: name of the winning player (None if draw/timeout)

    Returns:
        List of labeled row dicts suitable for ML training
    """
    events = parse_events(raw_events)
    players = infer_players(events)
    if not players:
        return []

    engine = GameEngine(players)
    rows = []

    i = 0
    while i < len(events):
        event = events[i]

        # Check if this event creates a challenge opportunity
        opportunity = _extract_challenge_opportunity(event, engine)

        if opportunity is not None:
            # Was this claim challenged? Look ahead for a ChallengeEvent
            challenge_event = None
            for j in range(i + 1, min(i + 3, len(events))):
                if isinstance(events[j], ChallengeEvent):
                    challenge_event = events[j]
                    break

            was_challenged = int(challenge_event is not None)

            # Outcome: track influence differential for the "would-be challenger"
            # within the next OUTCOME_WINDOW events
            challenger_candidates = [
                name
                for name in players
                if name != opportunity["claimant"]
                and int(
                    engine.public_state.players.get(
                        name,
                        type("x", (), {"influence_alive": 0})(),
                    ).influence_alive
                )
                > 0
            ]

            outcome_positive = _assess_outcome(
                events, i, challenge_event, challenger_candidates, winner, players
            )

            # Compute label: was challenging the RIGHT decision?
            # If the claim was a bluff (challenge won), challenging = optimal
            # If the claim was true (challenge lost), not challenging = optimal
            if challenge_event is not None:
                challenge_won = challenge_event.result == "win"
                label_optimal = int(challenge_won)
            else:
                # No challenge happened; estimate optimality from deck pressure
                remaining = int(opportunity.get("remaining_copies", 3))
                # If 0 copies remain, challenging would have been guaranteed correct
                label_optimal = int(remaining <= 0)

            # Belief features at time of opportunity
            probs = engine.belief_state.probabilities.get(opportunity["claimant"], {})
            role = opportunity["role"]

            row = {
                "game_event_index": i,
                "claimant": opportunity["claimant"],
                "claimed_role": role.value if hasattr(role, "value") else str(role),
                "context": opportunity["context"],
                "remaining_copies": int(opportunity.get("remaining_copies", 3)),
                "was_challenged": was_challenged,
                "label_optimal_challenge": label_optimal,
                "label_outcome_positive": int(outcome_positive),
                "f_belief_p_role": float(probs.get(role, 0.0)),
                "f_players_alive": sum(
                    1 for s in engine.public_state.players.values() if int(s.influence_alive) > 0
                ),
                "f_claimant_coins": int(
                    engine.public_state.players.get(
                        opportunity["claimant"],
                        type("x", (), {"coins": 0})(),
                    ).coins
                ),
                "winner": winner or "",
            }
            rows.append(row)

        engine.apply_event(event)
        i += 1

    return rows


def _extract_challenge_opportunity(event, engine) -> dict | None:
    """
    If the event is an action or block that involves a role claim,
    return opportunity metadata. Otherwise return None.
    """
    if isinstance(event, ActionEvent):
        entry = ACTION_ROLE_MULTIPLIERS.get(event.action_name)
        if entry:
            role, _ = entry
            remaining = max(0, 3 - int(engine.public_state.revealed_dead.get(role, 0)))
            return {
                "claimant": event.actor,
                "role": role,
                "context": "action",
                "action_name": event.action_name,
                "remaining_copies": remaining,
            }
    elif isinstance(event, BlockEvent):
        pairs = BLOCK_MULTIPLIERS.get(event.blocked_action, [])
        if pairs:
            role, _ = pairs[0]
            remaining = max(0, 3 - int(engine.public_state.revealed_dead.get(role, 0)))
            return {
                "claimant": event.blocker,
                "role": role,
                "context": "block",
                "blocked_action": event.blocked_action,
                "remaining_copies": remaining,
            }
    return None


def _assess_outcome(
    events, opportunity_index, challenge_event, challenger_candidates, winner, players
) -> bool:
    """
    Within OUTCOME_WINDOW events after the opportunity, compute whether
    the actual decision (challenge or no challenge) was net-positive.

    Simple heuristic:
    - If challenged and won: outcome = positive (opponent lost influence)
    - If challenged and lost: outcome = negative (we lost influence)
    - If not challenged and opponent was bluffing (revealed later): negative (missed it)
    - If not challenged and opponent was honest: positive (avoided bad challenge)
    """
    if challenge_event is not None:
        # Challenge happened — positive if the challenger won
        return challenge_event.result == "win"

    # No challenge — check if the claimant was later caught bluffing this role
    # within OUTCOME_WINDOW events
    window_end = min(opportunity_index + OUTCOME_WINDOW, len(events))
    for event in events[opportunity_index + 1 : window_end]:
        if isinstance(event, ChallengeEvent):
            # Someone else challenged and won → we missed the same opportunity
            if event.result == "win":
                return False
    # No evidence of bluff → not challenging was probably fine
    return True
