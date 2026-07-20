from __future__ import annotations

from coup.models import Role

ACTION_ROLE_MULTIPLIERS = {
    "Tax": (Role.DUKE, 2.0),
    "Assassinate": (Role.ASSASSIN, 2.0),
    "Steal": (Role.CAPTAIN, 2.0),
    "Exchange": (Role.AMBASSADOR, 2.0),
}

BLOCK_MULTIPLIERS = {
    "Foreign Aid": [(Role.DUKE, 2.0)],
    "Assassination": [(Role.CONTESSA, 2.0)],
    "Steal": [(Role.CAPTAIN, 1.6), (Role.AMBASSADOR, 1.6)],
}
# Bayesian likelihood ratios for belief updates.
# Ratio = P(makes_claim | has_role) / P(makes_claim | lacks_role)
# Honest rate ~0.85, bluff rate ~0.15 -> ratio = 5.67
# Challenge win  -> ~10x confirmation  (near-certain evidence)
# Challenge loss -> ~0.05x (near-certain disconfirmation)
ACTION_LIKELIHOOD_RATIOS = {
    "Tax": ("Duke", 5.7),
    "Assassinate": ("Assassin", 5.7),
    "Steal": ("Captain", 5.7),
    "Exchange": ("Ambassador", 5.7),
}

BLOCK_LIKELIHOOD_RATIOS = {
    "Foreign Aid": [("Duke", 5.7)],
    "Assassination": [("Contessa", 5.7)],
    "Steal": [("Captain", 4.5), ("Ambassador", 4.5)],
}

CHALLENGE_WIN_RATIO = 10.0
CHALLENGE_LOSE_RATIO = 0.05


def action_claim_role(action_name):
    entry = ACTION_ROLE_MULTIPLIERS.get(action_name)
    if entry is None:
        return None
    return entry[0]


def block_claim_roles(blocked_action):
    return [role for role, _ in BLOCK_MULTIPLIERS.get(blocked_action, [])]
