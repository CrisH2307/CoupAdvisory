from __future__ import annotations

from coup.models import ActionEvent, CoinChangeEvent, InfluenceLostEvent, RevealEvent


def validate_event(public_state, event):
    errors = []

    if isinstance(event, CoinChangeEvent):
        if event.player not in public_state.players:
            errors.append(f"Unknown player: {event.player}")
        else:
            coins = int(public_state.players[event.player].coins)
            if coins + int(event.delta) < 0:
                errors.append("Coin count cannot go negative")

    if isinstance(event, ActionEvent):
        if event.actor not in public_state.players:
            errors.append(f"Unknown player: {event.actor}")
            return errors

        if event.action_name in {"Steal", "Assassinate", "Coup"} and not event.target:
            errors.append(f"{event.action_name} requires a target")

        if event.target and event.target not in public_state.players:
            errors.append(f"Unknown target: {event.target}")
        if event.target and event.target == event.actor:
            errors.append("Action target cannot be the actor")

        actor_state = public_state.players[event.actor]
        actor_coins = int(actor_state.coins)

        if event.action_name == "Coup" and actor_coins < 7:
            errors.append("Coup requires 7 coins")

        if event.action_name == "Assassinate" and actor_coins < 3:
            errors.append("Assassinate requires 3 coins")

        if actor_coins >= 10 and event.action_name != "Coup":
            errors.append("Player with 10+ coins must Coup")

    if isinstance(event, (RevealEvent, InfluenceLostEvent)):
        if event.player not in public_state.players:
            errors.append(f"Unknown player: {event.player}")
        elif int(public_state.players[event.player].influence_alive) <= 0:
            errors.append("Player has no influence remaining")

    return errors


def validate_events(public_state, events):
    errors = []
    for event in events:
        errors.extend(validate_event(public_state, event))
    return errors
