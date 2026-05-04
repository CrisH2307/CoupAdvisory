from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from coup.models import Role


class BotStyle(str, Enum):
    HONEST = "honest"
    BLUFFER = "bluffer"
    AGGRESSIVE = "aggressive"
    CAUTIOUS = "cautious"
    RULE_BASED = "rule_based"
    BELIEF_EV = "belief_ev"
    PRESSURE = "pressure"


@dataclass
class ActionChoice:
    action_name: str
    target: str | None = None


@dataclass
class BotPlayerView:
    name: str
    coins: int
    influence_alive: int
    hidden_roles: list[Role]
    revealed_roles: list[Role]

    def has_role(self, role):
        return role in self.hidden_roles


def clamp(value, lower=0.0, upper=1.0):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


class _BaseBot:
    style = BotStyle.HONEST
    bluff_rate = 0.0
    base_challenge_rate = 0.1

    def choose_action(self, me, opponents, rng):
        if me.coins >= 10 and opponents:
            return ActionChoice("Coup", target=self._pick_target(opponents, rng))

        if me.coins >= 7 and opponents and rng.random() < 0.55:
            return ActionChoice("Coup", target=self._pick_target(opponents, rng))

        if me.coins >= 3 and opponents and self._can_claim(me, Role.ASSASSIN, rng):
            if rng.random() < self._assassinate_weight():
                return ActionChoice("Assassinate", target=self._pick_target(opponents, rng))

        steal_target = self._pick_steal_target(opponents, rng)
        if steal_target and self._can_claim(me, Role.CAPTAIN, rng):
            if rng.random() < self._steal_weight():
                return ActionChoice("Steal", target=steal_target)

        if self._can_claim(me, Role.DUKE, rng) and rng.random() < self._tax_weight():
            return ActionChoice("Tax")

        if self._can_claim(me, Role.AMBASSADOR, rng) and rng.random() < self._exchange_weight():
            return ActionChoice("Exchange")

        if rng.random() < self._foreign_aid_weight():
            return ActionChoice("Foreign Aid")
        return ActionChoice("Income")

    def choose_challenge(
        self,
        *,
        claimant,
        claimed_role,
        remaining_copies,
        context,
        blocked_action,
        me,
        rng,
    ):
        del claimant, claimed_role, context, blocked_action, me

        rate = float(self.base_challenge_rate)
        if remaining_copies <= 0:
            rate += 0.80
        elif remaining_copies == 1:
            rate += 0.35
        elif remaining_copies == 2:
            rate += 0.10

        return rng.random() < clamp(rate)

    def choose_block(self, *, blocked_action, allowed_roles, me, rng):
        del blocked_action
        owned = [role for role in allowed_roles if me.has_role(role)]
        if owned:
            return rng.choice(owned)
        return None

    def _can_claim(self, me, role, rng):
        return me.has_role(role) or rng.random() < self.bluff_rate

    def _pick_target(self, opponents, rng):
        if not opponents:
            return None
        ranked = sorted(opponents, key=lambda op: (op.coins, op.influence_alive), reverse=True)
        top = ranked[: min(2, len(ranked))]
        return rng.choice(top).name

    def _pick_steal_target(self, opponents, rng):
        viable = [op for op in opponents if op.coins > 0]
        if not viable:
            return None
        ranked = sorted(viable, key=lambda op: op.coins, reverse=True)
        top = ranked[: min(2, len(ranked))]
        return rng.choice(top).name

    def _assassinate_weight(self):
        return 0.35

    def _steal_weight(self):
        return 0.40

    def _tax_weight(self):
        return 0.45

    def _exchange_weight(self):
        return 0.20

    def _foreign_aid_weight(self):
        return 0.35


class _HonestBot(_BaseBot):
    style = BotStyle.HONEST
    bluff_rate = 0.0
    base_challenge_rate = 0.06

    def _assassinate_weight(self):
        return 0.30

    def _steal_weight(self):
        return 0.35

    def _tax_weight(self):
        return 0.55

    def _exchange_weight(self):
        return 0.25

    def _foreign_aid_weight(self):
        return 0.30


class _BlufferBot(_BaseBot):
    style = BotStyle.BLUFFER
    bluff_rate = 0.55
    base_challenge_rate = 0.14

    def choose_block(self, *, blocked_action, allowed_roles, me, rng):
        owned = [role for role in allowed_roles if me.has_role(role)]
        if owned:
            return rng.choice(owned)
        if rng.random() < 0.45:
            return rng.choice(allowed_roles)
        return None

    def _assassinate_weight(self):
        return 0.45

    def _steal_weight(self):
        return 0.55

    def _tax_weight(self):
        return 0.60

    def _exchange_weight(self):
        return 0.30

    def _foreign_aid_weight(self):
        return 0.25


class _AggressiveBot(_BaseBot):
    style = BotStyle.AGGRESSIVE
    bluff_rate = 0.25
    base_challenge_rate = 0.28

    def _assassinate_weight(self):
        return 0.70

    def _steal_weight(self):
        return 0.70

    def _tax_weight(self):
        return 0.25

    def _exchange_weight(self):
        return 0.10

    def _foreign_aid_weight(self):
        return 0.15


class _CautiousBot(_BaseBot):
    style = BotStyle.CAUTIOUS
    bluff_rate = 0.08
    base_challenge_rate = 0.03

    def choose_block(self, *, blocked_action, allowed_roles, me, rng):
        owned = [role for role in allowed_roles if me.has_role(role)]
        if owned and rng.random() < 0.85:
            return rng.choice(owned)
        if not owned and rng.random() < 0.04:
            return rng.choice(allowed_roles)
        return None

    def _assassinate_weight(self):
        return 0.15

    def _steal_weight(self):
        return 0.25

    def _tax_weight(self):
        return 0.35

    def _exchange_weight(self):
        return 0.30

    def _foreign_aid_weight(self):
        return 0.50


class _RuleBasedBot(_BaseBot):
    style = BotStyle.RULE_BASED
    bluff_rate = 0.06
    base_challenge_rate = 0.10

    def choose_action(self, me, opponents, rng):
        if me.coins >= 10 and opponents:
            return ActionChoice("Coup", target=self._pick_target(opponents, rng))
        if me.coins >= 7 and opponents:
            return ActionChoice("Coup", target=self._pick_target(opponents, rng))
        if me.coins >= 3 and opponents and len(opponents) <= 2 and self._can_claim(me, Role.ASSASSIN, rng):
            return ActionChoice("Assassinate", target=self._pick_target(opponents, rng))

        steal_target = self._pick_steal_target(opponents, rng)
        if steal_target and self._can_claim(me, Role.CAPTAIN, rng):
            return ActionChoice("Steal", target=steal_target)

        if self._can_claim(me, Role.DUKE, rng):
            return ActionChoice("Tax")
        if me.influence_alive <= 1 and self._can_claim(me, Role.AMBASSADOR, rng):
            return ActionChoice("Exchange")
        if me.coins <= 1:
            return ActionChoice("Foreign Aid")
        return ActionChoice("Income")

    def choose_challenge(
        self,
        *,
        claimant,
        claimed_role,
        remaining_copies,
        context,
        blocked_action,
        me,
        rng,
    ):
        del claimant, claimed_role, context, rng
        if remaining_copies <= 0:
            return True
        if blocked_action == "Assassination" and me.influence_alive <= 1 and remaining_copies <= 1:
            return True
        return False

    def choose_block(self, *, blocked_action, allowed_roles, me, rng):
        del blocked_action, rng
        owned = [role for role in allowed_roles if me.has_role(role)]
        if owned:
            return owned[0]
        return None


class _BeliefEVBot(_BaseBot):
    style = BotStyle.BELIEF_EV
    bluff_rate = 0.15
    base_challenge_rate = 0.16

    def choose_action(self, me, opponents, rng):
        if me.coins >= 10 and opponents:
            return ActionChoice("Coup", target=self._pick_target(opponents, rng))

        avg_opp_coins = 0.0
        if opponents:
            avg_opp_coins = sum(op.coins for op in opponents) / len(opponents)
        steal_target = self._pick_steal_target(opponents, rng)

        scores = {"Income": 1.0, "Foreign Aid": 1.4, "Tax": 2.0, "Exchange": 0.9}
        if self._can_claim(me, Role.DUKE, rng):
            scores["Tax"] += 0.8
        if self._can_claim(me, Role.CAPTAIN, rng):
            scores["Steal"] = min(2.0, max(0.0, avg_opp_coins)) * 0.9
        if me.coins >= 3 and self._can_claim(me, Role.ASSASSIN, rng):
            scores["Assassinate"] = 1.6 if len(opponents) <= 2 else 1.2
        if me.coins >= 7:
            scores["Coup"] = 2.1

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for action_name, _score in ordered:
            if action_name == "Steal" and steal_target:
                return ActionChoice("Steal", target=steal_target)
            if action_name in {"Assassinate", "Coup"} and opponents:
                return ActionChoice(action_name, target=self._pick_target(opponents, rng))
            if action_name in {"Income", "Foreign Aid", "Tax", "Exchange"}:
                return ActionChoice(action_name)
        return ActionChoice("Income")

    def choose_challenge(
        self,
        *,
        claimant,
        claimed_role,
        remaining_copies,
        context,
        blocked_action,
        me,
        rng,
    ):
        del claimant, claimed_role, rng
        p_truth = clamp(float(remaining_copies) / 3.0)
        threshold = 0.22
        if context == "block":
            threshold = 0.28
        if blocked_action == "Assassination" and me.influence_alive <= 1:
            threshold = 0.45
        return p_truth < threshold

    def choose_block(self, *, blocked_action, allowed_roles, me, rng):
        del blocked_action, rng
        owned = [role for role in allowed_roles if me.has_role(role)]
        if owned:
            return owned[0]
        if me.influence_alive <= 1 and allowed_roles:
            return allowed_roles[0]
        return None


class _PressureBot(_BaseBot):
    style = BotStyle.PRESSURE
    bluff_rate = 0.20
    base_challenge_rate = 0.18

    def choose_action(self, me, opponents, rng):
        if me.coins >= 10 and opponents:
            return ActionChoice("Coup", target=self._pick_target(opponents, rng))
        if len(opponents) <= 1:
            if me.coins >= 7:
                return ActionChoice("Coup", target=self._pick_target(opponents, rng))
            if me.coins >= 3 and self._can_claim(me, Role.ASSASSIN, rng):
                return ActionChoice("Assassinate", target=self._pick_target(opponents, rng))

        high_coin_targets = sorted(opponents, key=lambda op: op.coins, reverse=True) if opponents else []
        if me.coins >= 7 and high_coin_targets and high_coin_targets[0].coins >= 7:
            return ActionChoice("Coup", target=high_coin_targets[0].name)
        if me.coins >= 3 and opponents and (me.influence_alive <= 1 or high_coin_targets[0].coins >= 5):
            if self._can_claim(me, Role.ASSASSIN, rng):
                return ActionChoice("Assassinate", target=self._pick_target(opponents, rng))

        steal_target = self._pick_steal_target(opponents, rng)
        if steal_target and self._can_claim(me, Role.CAPTAIN, rng):
            return ActionChoice("Steal", target=steal_target)

        if me.coins <= 1:
            return ActionChoice("Foreign Aid")
        if self._can_claim(me, Role.DUKE, rng):
            return ActionChoice("Tax")
        return ActionChoice("Income")

    def choose_challenge(
        self,
        *,
        claimant,
        claimed_role,
        remaining_copies,
        context,
        blocked_action,
        me,
        rng,
    ):
        del claimant, claimed_role, rng
        if remaining_copies <= 0:
            return True
        if blocked_action == "Assassination" and me.influence_alive <= 1 and remaining_copies <= 1:
            return True
        if context == "action" and me.influence_alive <= 1 and remaining_copies <= 1:
            return True
        return False

    def choose_block(self, *, blocked_action, allowed_roles, me, rng):
        del blocked_action
        owned = [role for role in allowed_roles if me.has_role(role)]
        if owned:
            return rng.choice(owned)
        if me.influence_alive <= 1 and rng.random() < 0.35:
            return rng.choice(allowed_roles)
        return None


def create_bot(style):
    if style is None:
        style = BotStyle.HONEST.value
    normalized = str(style).strip().lower()
    if normalized == BotStyle.HONEST.value:
        return _HonestBot()
    if normalized == BotStyle.BLUFFER.value:
        return _BlufferBot()
    if normalized == BotStyle.AGGRESSIVE.value:
        return _AggressiveBot()
    if normalized == BotStyle.CAUTIOUS.value:
        return _CautiousBot()
    if normalized == BotStyle.RULE_BASED.value:
        return _RuleBasedBot()
    if normalized == BotStyle.BELIEF_EV.value:
        return _BeliefEVBot()
    if normalized == BotStyle.PRESSURE.value:
        return _PressureBot()
    raise ValueError(f"Unknown bot style: {style}")


def default_bot_mix(players):
    base = [
        BotStyle.HONEST.value,
        BotStyle.BLUFFER.value,
        BotStyle.AGGRESSIVE.value,
        BotStyle.CAUTIOUS.value,
    ]
    mix = []
    for index in range(players):
        mix.append(base[index % len(base)])
    return mix


def normalize_bot_mix(players, bot_mix):
    if bot_mix is None:
        return default_bot_mix(players)

    if isinstance(bot_mix, str):
        provided = [part.strip().lower() for part in bot_mix.split(",") if part.strip()]
    else:
        provided = [str(part).strip().lower() for part in bot_mix if str(part).strip()]

    if not provided:
        return default_bot_mix(players)

    normalized = []
    for index in range(players):
        style = provided[index % len(provided)]
        if style not in {s.value for s in BotStyle}:
            raise ValueError(f"Unknown bot style: {style}")
        normalized.append(style)
    return normalized
