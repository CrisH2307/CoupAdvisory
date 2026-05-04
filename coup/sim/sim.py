from __future__ import annotations

import random
from dataclasses import dataclass, field

from coup.models import ActionEvent, BlockEvent, ChallengeEvent, RevealEvent, Role
from coup.sim.bots import BotPlayerView, create_bot, normalize_bot_mix
from coup.sim.utils import build_deck, draw_cards, remaining_copies

_ACTION_CLAIMS = {
    "Tax": Role.DUKE,
    "Steal": Role.CAPTAIN,
    "Assassinate": Role.ASSASSIN,
    "Exchange": Role.AMBASSADOR,
}


@dataclass
class SimPlayer:
    name: str
    bot: object
    influence: list[Role]
    coins: int = 2
    revealed: list[Role] = field(default_factory=list)

    @property
    def influence_alive(self):
        return len(self.influence)

    @property
    def is_alive(self):
        return self.influence_alive > 0

    def view(self):
        return BotPlayerView(
            name=self.name,
            coins=int(self.coins),
            influence_alive=int(self.influence_alive),
            hidden_roles=list(self.influence),
            revealed_roles=list(self.revealed),
        )


@dataclass
class SimGameResult:
    players: list[str]
    events: list[dict]
    winner: str
    turns: int
    bot_styles: list[str] = field(default_factory=list)


class _SimState:
    def __init__(self, players, deck, rng):
        self.players = players
        self.deck = deck
        self.rng = rng
        self.events = []
        self.turns = 0

    def alive_players(self):
        return [name for name, player in self.players.items() if player.is_alive]

    def alive_opponents(self, actor_name):
        return [
            name
            for name, player in self.players.items()
            if name != actor_name and player.is_alive
        ]

    def apply_coin_delta(self, player_name, delta):
        if player_name not in self.players:
            return
        delta = int(delta)
        if delta == 0:
            return
        player = self.players[player_name]
        player.coins = max(0, int(player.coins) + delta)
        self.events.append(
            {
                "type": "coin_change",
                "player": player_name,
                "delta": delta,
            }
        )

    def revealed_dead(self):
        counts = {role: 0 for role in Role}
        for player in self.players.values():
            for role in player.revealed:
                counts[role] += 1
        return counts

    def lose_influence(self, player_name):
        player = self.players.get(player_name)
        if player is None or not player.is_alive:
            return None

        index = self.rng.randrange(len(player.influence))
        revealed_role = player.influence.pop(index)
        player.revealed.append(revealed_role)

        self.events.append(
            RevealEvent(player=player_name, revealed_role=revealed_role).model_dump(mode="json")
        )
        return revealed_role

    def run_exchange(self, player_name):
        player = self.players.get(player_name)
        if player is None or not player.is_alive:
            return

        keep_count = len(player.influence)
        drawn = draw_cards(self.deck, 2, self.rng)
        pool = list(player.influence) + list(drawn)
        if not pool:
            return

        self.rng.shuffle(pool)
        keep = pool[:keep_count]
        returned = pool[keep_count:]

        player.influence = keep
        self.deck.extend(returned)
        self.rng.shuffle(self.deck)

    def should_anyone_challenge(
        self,
        *,
        claimant,
        claimed_role,
        candidate_names,
        context,
        blocked_action=None,
    ):
        candidates = [
            name for name in candidate_names if name in self.players and self.players[name].is_alive
        ]
        self.rng.shuffle(candidates)
        copies_left = remaining_copies(self.revealed_dead(), claimed_role)

        for candidate in candidates:
            bot = self.players[candidate].bot
            me = self.players[candidate].view()
            if bot.choose_challenge(
                claimant=claimant,
                claimed_role=claimed_role,
                remaining_copies=copies_left,
                context=context,
                blocked_action=blocked_action,
                me=me,
                rng=self.rng,
            ):
                return candidate
        return None

    def resolve_challenge(self, *, challenger, challenged, claimed_role):
        challenged_player = self.players.get(challenged)
        challenger_player = self.players.get(challenger)
        if challenged_player is None or challenger_player is None:
            return False
        if not challenged_player.is_alive or not challenger_player.is_alive:
            return False

        challenged_has_role = claimed_role in challenged_player.influence
        loser = challenger if challenged_has_role else challenged
        loser_player = self.players[loser]

        preview_role = None
        if loser_player.influence:
            preview_role = self.rng.choice(loser_player.influence)

        self.events.append(
            ChallengeEvent(
                challenger=challenger,
                challenged=challenged,
                claimed_role=claimed_role,
                result="win" if challenged_has_role else "lose",
                revealed_role=preview_role,
            ).model_dump(mode="json")
        )
        self.lose_influence(loser)
        return challenged_has_role


def simulate_games(games, *, players=4, seed=None, bot_mix=None, max_turns=200):
    results = []
    start_seed = 0 if seed is None else int(seed)
    for index in range(int(games)):
        game_seed = start_seed + index
        results.append(
            simulate_game(
                players=players,
                seed=game_seed,
                bot_mix=bot_mix,
                max_turns=max_turns,
            )
        )
    return results


def simulate_game(*, players=4, seed=None, bot_mix=None, max_turns=200):
    players = int(players)
    max_turns = int(max_turns)
    if players < 2 or players > 6:
        raise ValueError("players must be between 2 and 6")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")

    rng = random.Random(seed)
    player_names = [f"P{index + 1}" for index in range(players)]
    styles = normalize_bot_mix(players, bot_mix)

    deck = build_deck()
    rng.shuffle(deck)

    sim_players = {}
    for name, style in zip(player_names, styles):
        sim_players[name] = SimPlayer(
            name=name,
            bot=create_bot(style),
            influence=draw_cards(deck, 2, rng),
            coins=2,
        )

    state = _SimState(sim_players, deck, rng)

    turn_index = 0
    while len(state.alive_players()) > 1 and state.turns < max_turns:
        actor_name = player_names[turn_index % len(player_names)]
        turn_index += 1
        if not state.players[actor_name].is_alive:
            continue
        _play_turn(state, actor_name)
        state.turns += 1

    alive = state.alive_players()
    if alive:
        winner = alive[0]
    else:
        winner = _fallback_winner(state, player_names)

    return SimGameResult(
        players=player_names,
        events=state.events,
        winner=winner,
        turns=state.turns,
        bot_styles=list(styles),
    )


def _fallback_winner(state, player_names):
    ranking = sorted(
        player_names,
        key=lambda name: (
            state.players[name].influence_alive,
            state.players[name].coins,
        ),
        reverse=True,
    )
    return ranking[0]


def _play_turn(state, actor_name):
    actor = state.players[actor_name]
    if not actor.is_alive:
        return

    opponents = state.alive_opponents(actor_name)
    if not opponents:
        return

    opponent_views = [state.players[name].view() for name in opponents]
    decision = actor.bot.choose_action(actor.view(), opponent_views, state.rng)

    action_name = str(decision.action_name)
    target = decision.target

    if actor.coins >= 10:
        action_name = "Coup"

    if action_name in {"Coup", "Steal", "Assassinate"}:
        if target not in opponents:
            target = state.rng.choice(opponents)

    if action_name == "Coup" and actor.coins < 7:
        action_name = "Income"
        target = None

    if action_name == "Assassinate" and actor.coins < 3:
        action_name = "Income"
        target = None

    if action_name not in {"Coup", "Steal", "Assassinate"}:
        target = None

    state.events.append(
        ActionEvent(actor=actor_name, action_name=action_name, target=target).model_dump(mode="json")
    )

    if action_name == "Income":
        state.apply_coin_delta(actor_name, 1)
        return

    if action_name == "Foreign Aid":
        if _resolve_foreign_aid_block(state, actor_name):
            return
        state.apply_coin_delta(actor_name, 2)
        return

    if action_name == "Coup":
        state.apply_coin_delta(actor_name, -7)
        if target and state.players[target].is_alive:
            state.lose_influence(target)
        return

    if action_name == "Assassinate":
        state.apply_coin_delta(actor_name, -3)

    claimed_role = _ACTION_CLAIMS.get(action_name)
    if claimed_role is not None:
        claim_ok = _resolve_claim_challenge(
            state,
            claimant=actor_name,
            claimed_role=claimed_role,
            candidates=opponents,
            context="action",
        )
        if not claim_ok:
            return

    if action_name == "Tax":
        state.apply_coin_delta(actor_name, 3)
        return

    if action_name == "Steal" and target and state.players[target].is_alive:
        if _resolve_target_block(state, actor_name, target, "Steal"):
            return
        if not state.players[target].is_alive:
            return

        amount = min(2, state.players[target].coins)
        state.apply_coin_delta(target, -amount)
        state.apply_coin_delta(actor_name, amount)
        return

    if action_name == "Assassinate" and target and state.players[target].is_alive:
        if _resolve_target_block(state, actor_name, target, "Assassination"):
            return
        if state.players[target].is_alive:
            state.lose_influence(target)
        return

    if action_name == "Exchange":
        state.run_exchange(actor_name)


def _resolve_claim_challenge(state, *, claimant, claimed_role, candidates, context, blocked_action=None):
    challenger = state.should_anyone_challenge(
        claimant=claimant,
        claimed_role=claimed_role,
        candidate_names=candidates,
        context=context,
        blocked_action=blocked_action,
    )
    if challenger is None:
        return True

    return state.resolve_challenge(
        challenger=challenger,
        challenged=claimant,
        claimed_role=claimed_role,
    )


def _resolve_foreign_aid_block(state, actor_name):
    candidates = state.alive_opponents(actor_name)
    state.rng.shuffle(candidates)

    for blocker_name in candidates:
        blocker = state.players[blocker_name]
        if not blocker.is_alive:
            continue

        role = blocker.bot.choose_block(
            blocked_action="Foreign Aid",
            allowed_roles=[Role.DUKE],
            me=blocker.view(),
            rng=state.rng,
        )
        if role is None:
            continue

        state.events.append(
            BlockEvent(
                blocker=blocker_name,
                blocked_action="Foreign Aid",
                roles_claimed=[role],
            ).model_dump(mode="json")
        )

        actor = state.players[actor_name]
        copies_left = remaining_copies(state.revealed_dead(), role)
        should_challenge = actor.bot.choose_challenge(
            claimant=blocker_name,
            claimed_role=role,
            remaining_copies=copies_left,
            context="block",
            blocked_action="Foreign Aid",
            me=actor.view(),
            rng=state.rng,
        )

        if should_challenge:
            block_holds = state.resolve_challenge(
                challenger=actor_name,
                challenged=blocker_name,
                claimed_role=role,
            )
            return block_holds

        return True

    return False


def _resolve_target_block(state, actor_name, target_name, action_name):
    target = state.players[target_name]
    if not target.is_alive:
        return False

    if action_name == "Steal":
        allowed_roles = [Role.CAPTAIN, Role.AMBASSADOR]
        blocked_action = "Steal"
    else:
        allowed_roles = [Role.CONTESSA]
        blocked_action = "Assassination"

    role = target.bot.choose_block(
        blocked_action=blocked_action,
        allowed_roles=allowed_roles,
        me=target.view(),
        rng=state.rng,
    )
    if role is None:
        return False

    state.events.append(
        BlockEvent(
            blocker=target_name,
            blocked_action=blocked_action,
            roles_claimed=[role],
        ).model_dump(mode="json")
    )

    actor = state.players[actor_name]
    copies_left = remaining_copies(state.revealed_dead(), role)
    should_challenge = actor.bot.choose_challenge(
        claimant=target_name,
        claimed_role=role,
        remaining_copies=copies_left,
        context="block",
        blocked_action=blocked_action,
        me=actor.view(),
        rng=state.rng,
    )

    if should_challenge:
        block_holds = state.resolve_challenge(
            challenger=actor_name,
            challenged=target_name,
            claimed_role=role,
        )
        return block_holds

    return True
