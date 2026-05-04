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


def action_claim_role(action_name):
    entry = ACTION_ROLE_MULTIPLIERS.get(action_name)
    if entry is None:
        return None
    return entry[0]


def block_claim_roles(blocked_action):
    return [role for role, _ in BLOCK_MULTIPLIERS.get(blocked_action, [])]
