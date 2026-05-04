from __future__ import annotations

from collections import Counter

from coup.models import ActionEvent, BlockEvent, ChallengeEvent, parse_events


def compare_events(predicted_events, ground_truth_events):
    predicted = parse_events(predicted_events)
    truth = parse_events(ground_truth_events)

    pred_labels = [_event_label(event) for event in predicted]
    true_labels = [_event_label(event) for event in truth]

    pred_counter = Counter(pred_labels)
    true_counter = Counter(true_labels)

    matches = 0
    for label, count in pred_counter.items():
        matches += min(count, true_counter.get(label, 0))

    precision = matches / len(pred_labels) if pred_labels else 0.0
    recall = matches / len(true_labels) if true_labels else 0.0

    per_type = {}
    for label in sorted(set(true_counter) | set(pred_counter)):
        true_count = true_counter.get(label, 0)
        pred_count = pred_counter.get(label, 0)
        hit = min(true_count, pred_count)
        per_type[label] = {
            "true": true_count,
            "pred": pred_count,
            "matched": hit,
            "accuracy": (hit / true_count) if true_count else 0.0,
        }

    return {
        "precision": precision,
        "recall": recall,
        "per_event_type": per_type,
    }


def _event_label(event):
    if isinstance(event, ActionEvent):
        return f"action:{event.action_name}"
    if isinstance(event, BlockEvent):
        return f"block:{event.blocked_action}"
    if isinstance(event, ChallengeEvent):
        return f"challenge:{event.claimed_role.value}"
    return event.type
