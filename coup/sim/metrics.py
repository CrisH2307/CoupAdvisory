from __future__ import annotations

from dataclasses import dataclass, field

from coup.advisor import ClaimContext, recommend_challenge
from coup.engine import GameEngine
from coup.models import ActionEvent, BlockEvent, ChallengeEvent, InfluenceLostEvent, RevealEvent, parse_events
from coup.rules import action_claim_role, block_claim_roles


@dataclass
class GameMetrics:
    winner: str
    turns: int
    challenges: int
    challenge_accuracy: float
    advisor_agreement_rate: float
    advisor_claims_total: int
    advisor_recommended_challenges: int
    advisor_actual_challenges: int
    advisor_challenge_precision: float
    advisor_challenge_recall: float
    advisor_challenge_f1: float
    advisor_challenge_accuracy: float
    advisor_outcome_accuracy: float
    advisor_outcome_score: float
    reveal_probability_bins: dict[str, int] = field(default_factory=dict)

    @property
    def reveal_prob_histogram(self):
        # Backward-compatibility alias for older notebooks.
        return self.reveal_probability_bins


def calibration_bin_labels():
    labels = []
    for index in range(10):
        low = index / 10
        high = (index + 1) / 10
        labels.append(f"{low:.1f}-{high:.1f}")
    return labels


def build_trace(events, players, *, strict_no_duplicate_hand=False, advisor_style="Balanced"):
    parsed = parse_events(events)
    engine = GameEngine(players, strict_no_duplicate_hand=strict_no_duplicate_hand)

    trace_rows = []
    for index, event in enumerate(parsed):
        pre_advisor = _advisor_recommendations(engine, event, advisor_style=advisor_style)
        pre_state = engine.public_state.model_dump(mode="json")
        pre_belief = _belief_snapshot(engine)

        engine.apply_event(event)

        post_advisor = _advisor_recommendations(engine, event, advisor_style=advisor_style)
        post_state = engine.public_state.model_dump(mode="json")
        post_belief = _belief_snapshot(engine)

        trace_rows.append(
            {
                "event_index": index,
                "event": event.model_dump(mode="json"),
                "public_state_pre": pre_state,
                "belief_pre": pre_belief,
                "public_state_post": post_state,
                "belief_post": post_belief,
                "advisor_pre_event": pre_advisor,
                "advisor_post_event": post_advisor,
            }
        )

    return trace_rows


def evaluate_advisor_decisions(
    events,
    players,
    *,
    strict_no_duplicate_hand=False,
    advisor_policy="advisor",
    advisor_style="Balanced",
):
    parsed = parse_events(events)
    engine = GameEngine(players, strict_no_duplicate_hand=strict_no_duplicate_hand)

    claims_total = 0
    recommended = 0
    actual = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    outcome_total = 0
    outcome_correct = 0
    outcome_score_total = 0.0

    for index, event in enumerate(parsed):
        claims = _claims_from_event(event)
        if not claims:
            engine.apply_event(event)
            continue

        claim_outcomes = _find_claim_outcomes(parsed, index, claims)
        claim_recs = _advisor_recommendations(engine, event, advisor_style=advisor_style)

        for rec in claim_recs:
            key = (rec["claimant"], rec["role"])
            outcome = claim_outcomes.get(key)
            actual_challenge = bool(outcome and outcome.get("challenged"))
            actual_truth = None if outcome is None else outcome.get("claim_true")

            decision = _policy_decision(rec, advisor_policy)

            claims_total += 1
            if decision:
                recommended += 1
            if actual_challenge:
                actual += 1

            if decision and actual_challenge:
                tp += 1
            elif decision and not actual_challenge:
                fp += 1
            elif (not decision) and actual_challenge:
                fn += 1
            else:
                tn += 1

            if actual_truth is not None:
                outcome_total += 1
                predicted_truth = float(rec["p_truth"]) >= float(rec["threshold"])
                if predicted_truth == bool(actual_truth):
                    outcome_correct += 1
                    outcome_score_total += 1.0
                else:
                    outcome_score_total -= 1.0

        engine.apply_event(event)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    challenge_f1 = _safe_div(2 * precision * recall, precision + recall)
    challenge_accuracy = _safe_div(tp + tn, claims_total)
    outcome_accuracy = _safe_div(outcome_correct, outcome_total)
    outcome_score = _safe_div(outcome_score_total, outcome_total)

    return {
        "claims_total": float(claims_total),
        "recommended_challenges": float(recommended),
        "actual_challenges": float(actual),
        "challenge_precision": float(precision),
        "challenge_recall": float(recall),
        "challenge_f1": float(challenge_f1),
        "challenge_accuracy": float(challenge_accuracy),
        "outcome_accuracy": float(outcome_accuracy),
        "outcome_score": float(outcome_score),
    }


def summarize_game(
    *,
    events,
    players,
    winner,
    turns,
    strict_no_duplicate_hand=False,
    advisor_policy="advisor",
    advisor_style="Balanced",
):
    parsed = parse_events(events)
    challenge_events = [event for event in parsed if isinstance(event, ChallengeEvent)]
    challenges = len(challenge_events)
    correct_challenges = sum(1 for event in challenge_events if event.result == "lose")
    challenge_accuracy = _safe_div(correct_challenges, challenges)

    advisor = evaluate_advisor_decisions(
        events,
        players,
        strict_no_duplicate_hand=strict_no_duplicate_hand,
        advisor_policy=advisor_policy,
        advisor_style=advisor_style,
    )

    reveal_bins = _reveal_probability_histogram(
        events,
        players,
        strict_no_duplicate_hand=strict_no_duplicate_hand,
    )

    return GameMetrics(
        winner=winner,
        turns=int(turns),
        challenges=int(challenges),
        challenge_accuracy=float(challenge_accuracy),
        advisor_agreement_rate=float(advisor["challenge_accuracy"]),
        advisor_claims_total=int(advisor["claims_total"]),
        advisor_recommended_challenges=int(advisor["recommended_challenges"]),
        advisor_actual_challenges=int(advisor["actual_challenges"]),
        advisor_challenge_precision=float(advisor["challenge_precision"]),
        advisor_challenge_recall=float(advisor["challenge_recall"]),
        advisor_challenge_f1=float(advisor["challenge_f1"]),
        advisor_challenge_accuracy=float(advisor["challenge_accuracy"]),
        advisor_outcome_accuracy=float(advisor["outcome_accuracy"]),
        advisor_outcome_score=float(advisor["outcome_score"]),
        reveal_probability_bins=reveal_bins,
    )


def _belief_snapshot(engine):
    snapshot = {}
    for player, probs in engine.belief_state.probabilities.items():
        snapshot[player] = {role.value: float(value) for role, value in probs.items()}
    return snapshot


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


def _advisor_recommendations(engine, event, *, advisor_style="Balanced"):
    claims = _claims_from_event(event)
    if not claims:
        return []

    remaining = engine.remaining_copies()
    rows = []
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
            {
                "claimant": claim["claimant"],
                "role": claim["role"].value,
                "recommendation": rec.recommendation,
                "p_truth": float(rec.p_truth),
                "threshold": float(rec.threshold),
                "explanation": rec.explanation,
            }
        )
    return rows


def _find_claim_outcomes(parsed, claim_index, claims):
    outcomes = {
        (claim["claimant"], claim["role"].value): {
            "challenged": False,
            "claim_true": None,
        }
        for claim in claims
    }

    for index in range(claim_index + 1, len(parsed)):
        event = parsed[index]
        if isinstance(event, (ActionEvent, BlockEvent)):
            break
        if not isinstance(event, ChallengeEvent):
            continue

        key = (event.challenged, event.claimed_role.value)
        if key in outcomes:
            outcomes[key] = {
                "challenged": True,
                "claim_true": event.result == "win",
            }
    return outcomes


def _policy_decision(rec, policy):
    if policy == "always_challenge":
        return True
    if policy == "never_challenge":
        return False
    return rec["recommendation"] == "Challenge"


def _reveal_probability_histogram(events, players, *, strict_no_duplicate_hand=False):
    labels = calibration_bin_labels()
    bins = {label: 0 for label in labels}

    parsed = parse_events(events)
    engine = GameEngine(players, strict_no_duplicate_hand=strict_no_duplicate_hand)

    for event in parsed:
        if isinstance(event, (RevealEvent, InfluenceLostEvent)):
            probs = engine.belief_state.probabilities.get(event.player, {})
            probability = float(probs.get(event.revealed_role, 0.0))
            index = int(probability * 10)
            if index >= 10:
                index = 9
            if index < 0:
                index = 0
            bins[labels[index]] += 1
        engine.apply_event(event)

    return bins


def _safe_div(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
