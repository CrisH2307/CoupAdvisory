from __future__ import annotations

from coup.models import Role


def clamp(value, lower=0.0, upper=1.0):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def build_deck():
    deck = []
    for role in Role:
        deck.extend([role, role, role])
    return deck


def draw_cards(deck, count, rng):
    cards = []
    for _ in range(max(0, int(count))):
        if not deck:
            break
        index = rng.randrange(len(deck))
        cards.append(deck.pop(index))
    return cards


def remaining_copies(revealed_dead, role):
    revealed = int(revealed_dead.get(role, 0))
    return max(0, 3 - revealed)
