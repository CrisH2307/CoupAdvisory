from __future__ import annotations
"""
Tabular Counterfactual Regret Minimization (CFR) for the 2-player,
1-influence-each Coup endgame.

This finds Nash Equilibrium strategies for the endgame where:
- Player 1 has exactly 1 influence card (unknown to Player 2)
- Player 2 has exactly 1 influence card (unknown to Player 1)
- The game ends when either player loses their last influence

Key simplification: we abstract coin economy into discrete buckets
(0-2, 3-6, 7+) to keep the state space tractable.

Usage:
    from coup.research.cfr import CoupEndgameCFR, EndgameState
    cfr = CoupEndgameCFR(iterations=10_000)
    cfr.train()
    state = EndgameState(p1_role="Duke", p2_role="Assassin", p1_coin_bucket=1, p2_coin_bucket=1)
    strategy = cfr.get_strategy(state, player=1)
    print(strategy)  # {"Tax": 0.6, "Income": 0.4, ...}
"""
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal


ROLES = ["Duke", "Assassin", "Captain", "Ambassador", "Contessa"]

# Simplified coin buckets: 0=0-2 coins, 1=3-6 coins, 2=7+ coins
COIN_BUCKETS = [0, 1, 2]
COIN_BUCKET_LABELS = ["0-2", "3-6", "7+"]

# Actions available per coin bucket (simplified for 1-influence endgame)
ACTIONS_BY_BUCKET = {
    0: ["Income", "Foreign Aid", "Tax", "Exchange"],           # <3 coins
    1: ["Income", "Foreign Aid", "Tax", "Steal", "Assassinate", "Exchange"],  # 3-6 coins
    2: ["Coup"],                                                # 7+ coins: must Coup
}

# Role abilities (which roles can block/enable which actions)
ROLE_ACTIONS = {
    "Duke": ["Tax"],
    "Assassin": ["Assassinate"],
    "Captain": ["Steal"],
    "Ambassador": ["Exchange"],
    "Contessa": [],  # defensive only
}

ROLE_BLOCKS = {
    "Duke": ["Foreign Aid"],
    "Captain": ["Steal"],
    "Ambassador": ["Steal"],
    "Contessa": ["Assassination"],
}

# CFR control knobs
# Depth was previously too high for this abstract game and caused the
# first iteration to take extremely long due to repeated non-terminal loops.
MAX_CFR_DEPTH = 6


@dataclass(frozen=True)
class EndgameState:
    p1_role: str          # role name string
    p2_role: str          # role name string
    p1_coin_bucket: int   # 0=low, 1=mid, 2=high
    p2_coin_bucket: int
    p1_turn: bool = True  # True = P1 is acting

    def __str__(self) -> str:
        return (
            f"P1={self.p1_role}/{COIN_BUCKET_LABELS[self.p1_coin_bucket]} "
            f"P2={self.p2_role}/{COIN_BUCKET_LABELS[self.p2_coin_bucket]} "
            f"turn={'P1' if self.p1_turn else 'P2'}"
        )


@dataclass
class InfoSet:
    """
    Information set for one player — what they know:
    - Their own role and coin bucket
    - Opponent's coin bucket (visible)
    - Whose turn it is
    (Opponent's role is hidden)
    """

    own_role: str
    own_coin_bucket: int
    opp_coin_bucket: int
    is_my_turn: bool

    def key(self) -> tuple:
        return (self.own_role, self.own_coin_bucket, self.opp_coin_bucket, self.is_my_turn)


class CoupEndgameCFR:
    """
    Tabular CFR for 2-player 1-influence Coup endgame.

    Trains a Nash Equilibrium strategy by repeatedly sampling games
    and updating regret tables using the CFR algorithm.
    """

    def __init__(self, iterations: int = 5_000, seed: int = 42):
        self.iterations = iterations
        self.rng = random.Random(seed)

        # regret_sum[infoset_key][action] = cumulative counterfactual regret
        self.regret_sum: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # strategy_sum[infoset_key][action] = cumulative strategy (for averaging)
        self.strategy_sum: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        self._trained = False

    def train(self) -> None:
        """Run CFR for self.iterations iterations."""
        print(f"Training CFR for {self.iterations} iterations...", flush=True)
        start_ts = time.perf_counter()
        progress_step = max(1, min(10, self.iterations // 100))  # print every 10 iters (or denser for tiny runs)
        for i in range(1, self.iterations + 1):
            if i == 1 or i % progress_step == 0:
                elapsed = time.perf_counter() - start_ts
                pct = (i / self.iterations) * 100.0
                print(
                    f"[CFR] starting {i}/{self.iterations} ({pct:5.1f}%) "
                    f"elapsed={elapsed:6.2f}s",
                    flush=True,
                )
            # Sample a random card deal
            p1_role = self.rng.choice(ROLES)
            p2_role = self.rng.choice(ROLES)
            p1_coins = self.rng.choice(COIN_BUCKETS)
            p2_coins = self.rng.choice(COIN_BUCKETS)

            state = EndgameState(
                p1_role=p1_role,
                p2_role=p2_role,
                p1_coin_bucket=p1_coins,
                p2_coin_bucket=p2_coins,
                p1_turn=True,
            )
            # Run CFR from P1's perspective
            self._cfr(state, reach_p1=1.0, reach_p2=1.0, depth=0)
            if i % progress_step == 0 or i == self.iterations:
                elapsed = time.perf_counter() - start_ts
                per_iter = elapsed / i if i > 0 else 0.0
                remaining = self.iterations - i
                eta_seconds = remaining * per_iter
                pct = (i / self.iterations) * 100.0
                print(
                    f"[CFR] {i}/{self.iterations} ({pct:5.1f}%) "
                    f"elapsed={elapsed:6.2f}s eta={eta_seconds:6.2f}s"
                    ,
                    flush=True,
                )

        self._trained = True
        total_elapsed = time.perf_counter() - start_ts
        print(f"CFR training complete in {total_elapsed:.2f}s.", flush=True)

    def get_strategy(self, state: EndgameState, player: Literal[1, 2] = 1) -> dict[str, float]:
        """
        Return the average mixed strategy for the given state and player.
        The 'player' arg indicates which player's perspective (their role is known).
        """
        if player == 1:
            infoset = InfoSet(
                own_role=state.p1_role,
                own_coin_bucket=state.p1_coin_bucket,
                opp_coin_bucket=state.p2_coin_bucket,
                is_my_turn=state.p1_turn,
            )
        else:
            infoset = InfoSet(
                own_role=state.p2_role,
                own_coin_bucket=state.p2_coin_bucket,
                opp_coin_bucket=state.p1_coin_bucket,
                is_my_turn=not state.p1_turn,
            )

        key = infoset.key()
        strategy_sum = self.strategy_sum.get(key, {})
        actions = self._legal_actions(
            state.p1_coin_bucket if player == 1 else state.p2_coin_bucket
        )

        if not strategy_sum or sum(strategy_sum.values()) == 0:
            # Uniform fallback if not trained or no visits
            return {a: 1.0 / len(actions) for a in actions}

        total = sum(strategy_sum.get(a, 0.0) for a in actions)
        if total <= 0:
            return {a: 1.0 / len(actions) for a in actions}

        return {a: strategy_sum.get(a, 0.0) / total for a in actions}

    def _legal_actions(self, coin_bucket: int) -> list[str]:
        return list(ACTIONS_BY_BUCKET.get(coin_bucket, ["Income"]))

    def _cfr(self, state: EndgameState, reach_p1: float, reach_p2: float, depth: int) -> float:
        """
        Recursive CFR. Returns expected utility for the current acting player.
        Simplified: terminal states return +1 (win) or -1 (loss).
        """
        if depth > MAX_CFR_DEPTH:
            return 0.0  # depth limit to prevent infinite loops

        acting_role = state.p1_role if state.p1_turn else state.p2_role
        acting_coins = state.p1_coin_bucket if state.p1_turn else state.p2_coin_bucket
        actions = self._legal_actions(acting_coins)

        # Build information set key for the acting player
        if state.p1_turn:
            infoset_key = (acting_role, state.p1_coin_bucket, state.p2_coin_bucket, True)
            reach_self, reach_opp = reach_p1, reach_p2
        else:
            infoset_key = (acting_role, state.p2_coin_bucket, state.p1_coin_bucket, True)
            reach_self, reach_opp = reach_p2, reach_p1

        strategy = self._get_current_strategy(infoset_key, actions)

        # Compute utility for each action
        action_utils = {}
        node_util = 0.0
        for action in actions:
            next_state, terminal_util = self._transition(state, action)
            if terminal_util is not None:
                action_utils[action] = terminal_util
            else:
                if state.p1_turn:
                    action_utils[action] = -self._cfr(
                        next_state,
                        reach_p1 * strategy[action],
                        reach_p2,
                        depth + 1,
                    )
                else:
                    action_utils[action] = -self._cfr(
                        next_state,
                        reach_p1,
                        reach_p2 * strategy[action],
                        depth + 1,
                    )
            node_util += strategy[action] * action_utils[action]

        # Update regrets
        for action in actions:
            regret = action_utils[action] - node_util
            self.regret_sum[infoset_key][action] += reach_opp * regret
            self.strategy_sum[infoset_key][action] += reach_self * strategy[action]

        return node_util

    def _get_current_strategy(self, infoset_key: tuple, actions: list[str]) -> dict[str, float]:
        """Regret-matching: positive regrets proportional to strategy."""
        regrets = self.regret_sum.get(infoset_key, {})
        positive = {a: max(0.0, regrets.get(a, 0.0)) for a in actions}
        total = sum(positive.values())
        if total <= 0:
            return {a: 1.0 / len(actions) for a in actions}
        return {a: positive[a] / total for a in actions}

    def _transition(self, state: EndgameState, action: str) -> tuple[EndgameState | None, float | None]:
        """
        Apply action and return (next_state, terminal_utility) where:
        - terminal_utility = None if game continues
        - terminal_utility = +1 if acting player wins, -1 if acting player loses
        """
        p1_turn = state.p1_turn
        opp_role = state.p2_role if p1_turn else state.p1_role

        if action == "Coup":
            # Guaranteed: opponent loses influence → acting player wins
            return None, 1.0

        if action == "Assassinate":
            # Check if opponent can block with Contessa
            if opp_role == "Contessa":
                # Block succeeds (simplified: always block in endgame)
                # Acting player spent 3 coins; no kill
                new_p1_coins = max(0, state.p1_coin_bucket - 1) if p1_turn else state.p1_coin_bucket
                new_p2_coins = max(0, state.p2_coin_bucket - 1) if not p1_turn else state.p2_coin_bucket
            else:
                # No block: opponent eliminated → win
                return None, 1.0

            next_state = EndgameState(
                p1_role=state.p1_role,
                p2_role=state.p2_role,
                p1_coin_bucket=new_p1_coins,
                p2_coin_bucket=new_p2_coins,
                p1_turn=not p1_turn,
            )
            return next_state, None

        if action == "Income":
            new_p1 = min(2, state.p1_coin_bucket + 1) if p1_turn else state.p1_coin_bucket
            new_p2 = min(2, state.p2_coin_bucket + 1) if not p1_turn else state.p2_coin_bucket
            next_state = EndgameState(
                p1_role=state.p1_role,
                p2_role=state.p2_role,
                p1_coin_bucket=new_p1,
                p2_coin_bucket=new_p2,
                p1_turn=not p1_turn,
            )
            return next_state, None

        # Default: action changes coins, game continues with opponent's turn
        next_state = EndgameState(
            p1_role=state.p1_role,
            p2_role=state.p2_role,
            p1_coin_bucket=state.p1_coin_bucket,
            p2_coin_bucket=state.p2_coin_bucket,
            p1_turn=not p1_turn,
        )
        return next_state, None

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "regret_sum": dict(self.regret_sum),
                    "strategy_sum": dict(self.strategy_sum),
                    "iterations": self.iterations,
                },
                f,
            )
        print(f"CFR state saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CoupEndgameCFR":
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        cfr = cls(iterations=data["iterations"])
        cfr.regret_sum = defaultdict(lambda: defaultdict(float), data["regret_sum"])
        cfr.strategy_sum = defaultdict(lambda: defaultdict(float), data["strategy_sum"])
        cfr._trained = True
        return cfr
