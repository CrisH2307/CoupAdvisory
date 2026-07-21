from __future__ import annotations
"""
Evaluation utilities for the CFR endgame solver.
Run to inspect strategies for specific endgame scenarios.

Usage:
    python -m coup.research.cfr_eval
"""
from coup.research.cfr import COIN_BUCKET_LABELS, ROLES, CoupEndgameCFR, EndgameState


def print_strategy_table(cfr: CoupEndgameCFR) -> None:
    """Print the equilibrium strategy for each role pairing."""
    print("\n=== CFR Endgame Strategy Table ===")
    print(f"{'P1 Role':<12} {'P2 Role':<12} {'P1 Coins':<10} {'P2 Coins':<10} {'Best Action':<15} {'Prob'}")
    print("-" * 75)

    for p1_role in ROLES:
        for p2_role in ROLES:
            for p1_coins in [0, 1, 2]:
                for p2_coins in [0, 1, 2]:
                    state = EndgameState(
                        p1_role=p1_role,
                        p2_role=p2_role,
                        p1_coin_bucket=p1_coins,
                        p2_coin_bucket=p2_coins,
                        p1_turn=True,
                    )
                    strategy = cfr.get_strategy(state, player=1)
                    if not strategy:
                        continue
                    best_action = max(strategy, key=strategy.get)
                    best_prob = strategy[best_action]
                    if best_prob < 0.5:
                        continue  # only print dominant actions
                    print(
                        f"{p1_role:<12} {p2_role:<12} "
                        f"{COIN_BUCKET_LABELS[p1_coins]:<10} {COIN_BUCKET_LABELS[p2_coins]:<10} "
                        f"{best_action:<15} {best_prob:.2f}"
                    )


def main() -> None:
    print("Training CFR solver for 2-player 1-influence endgame...")
    cfr = CoupEndgameCFR(iterations=10_000, seed=42)
    cfr.train()
    print_strategy_table(cfr)

    # Spotlight: Duke vs Assassin with mid coins
    state = EndgameState(
        p1_role="Duke",
        p2_role="Assassin",
        p1_coin_bucket=1,
        p2_coin_bucket=1,
        p1_turn=True,
    )
    print(f"\nSpotlight — {state}")
    print("P1 (Duke) equilibrium strategy:")
    for action, prob in sorted(cfr.get_strategy(state, player=1).items(), key=lambda x: -x[1]):
        if prob > 0.01:
            print(f"  {action}: {prob:.3f}")


if __name__ == "__main__":
    main()
