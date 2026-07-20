from __future__ import annotations
"""
2-ply Expected Value calculator for Coup advisor decisions.

For each candidate action, computes:
  EV(action) = sum over likely opponent responses of:
      P(response) × value(state after action + response)

Value is expressed in "influence-adjusted coin units":
  - +1 coin gained = +1.0
  - Opponent loses 1 influence = +2.5 (scales with how close they are to death)
  - You lose 1 influence = -3.0 (scales with how many you have left)
"""
from coup.models import Role


# Value constants (tunable)
COIN_VALUE = 1.0
OPP_INFLUENCE_LOSS_VALUE = 2.5   # base value; doubled if it kills them
SELF_INFLUENCE_LOSS_COST = 3.0   # base cost; doubled if it kills you


def _self_influence_cost(my_influence: int) -> float:
    """Cost of losing one influence, scaled by how many you have left."""
    if my_influence <= 1:
        return SELF_INFLUENCE_LOSS_COST * 2.5  # game-ending loss
    return SELF_INFLUENCE_LOSS_COST


def _opp_influence_value(opp_influence: int) -> float:
    """Value of opponent losing one influence."""
    if opp_influence <= 1:
        return OPP_INFLUENCE_LOSS_VALUE * 2.0  # kills them
    return OPP_INFLUENCE_LOSS_VALUE


def ev_income() -> float:
    return COIN_VALUE * 1.0


def ev_foreign_aid(belief: dict, alive_opponents: list[str], my_influence: int) -> float:
    """
    EV of Foreign Aid considering possible Duke block.
    If blocked: EV of challenging the block (based on belief).
    """
    # Probability any opponent blocks with Duke
    p_block = max(
        float(belief.get(name, {}).get(Role.DUKE, 0.0))
        for name in alive_opponents
    ) if alive_opponents else 0.0

    # If not blocked: gain 2 coins
    ev_no_block = (1 - p_block) * (COIN_VALUE * 2)

    # If blocked: lose the Foreign Aid, no coins gained
    # We could challenge but that requires knowing if the Duke is real.
    # Conservative model: accept the block with prob p_block.
    ev_blocked = p_block * 0.0

    return ev_no_block + ev_blocked


def ev_tax(belief: dict, alive_opponents: list[str], my_duke_prob: float, my_influence: int) -> float:
    """
    EV of Tax (Duke claim).
    Risk: someone challenges. Payoff: +3 coins.
    """
    # Aggregate challenge probability: increases with number of skeptical opponents
    # Using the advisors's challenge_pressure heuristic concept here
    p_challenged_per_opp = max(0.0, (1.0 - my_duke_prob) * 0.35)
    p_not_challenged = (1.0 - p_challenged_per_opp) ** max(1, len(alive_opponents))

    # If not challenged: gain 3 coins
    ev_success = p_not_challenged * (COIN_VALUE * 3)

    # If challenged and you DON'T have Duke (bluffing): lose 1 influence
    p_challenged = 1.0 - p_not_challenged
    p_bluffing = 1.0 - my_duke_prob
    ev_challenge_loss = p_challenged * p_bluffing * (-_self_influence_cost(my_influence))

    # If challenged and you DO have Duke: challenger loses influence (+value to you)
    ev_challenge_win = p_challenged * my_duke_prob * _opp_influence_value(2)  # assume opp has 2

    return ev_success + ev_challenge_loss + ev_challenge_win


def ev_steal(
    belief: dict,
    target: str,
    target_coins: int,
    target_influence: int,
    my_captain_prob: float,
    my_influence: int,
) -> float:
    """
    EV of Steal from target.
    Opponent may block with Captain or Ambassador.
    """
    steal_amount = min(2, target_coins)
    if steal_amount == 0:
        return 0.0

    target_probs = belief.get(target, {})
    p_block_captain = float(target_probs.get(Role.CAPTAIN, 0.0))
    p_block_ambassador = float(target_probs.get(Role.AMBASSADOR, 0.0))
    p_block = min(0.95, p_block_captain + p_block_ambassador)

    # Challenge pressure on your Captain claim
    p_challenged = max(0.0, (1.0 - my_captain_prob) * 0.30)

    # Branch 1: action challenged (not blocked), you fail challenge
    ev_action_fail = p_challenged * (1 - my_captain_prob) * (-_self_influence_cost(my_influence))
    # Branch 2: action succeeds, no block
    ev_no_block = (1 - p_challenged) * (1 - p_block) * (COIN_VALUE * steal_amount)
    # Branch 3: blocked (block accepted)
    ev_blocked = (1 - p_challenged) * p_block * 0.0

    return ev_action_fail + ev_no_block + ev_blocked


def ev_assassinate(
    belief: dict,
    target: str,
    target_influence: int,
    my_assassin_prob: float,
    my_influence: int,
) -> float:
    """
    EV of Assassinate (costs 3 coins).
    Target may block with Contessa.
    """
    target_probs = belief.get(target, {})
    p_contessa = float(target_probs.get(Role.CONTESSA, 0.0))

    # Challenge pressure on your Assassin claim
    p_challenged = max(0.0, (1.0 - my_assassin_prob) * 0.30)

    coin_cost = -(COIN_VALUE * 3)

    # Branch 1: action challenged and you're bluffing
    ev_action_fail = p_challenged * (1 - my_assassin_prob) * (
        -_self_influence_cost(my_influence) + (COIN_VALUE * 3)  # coins refunded
    )
    # Branch 2: not challenged, target doesn't block
    ev_lands = (1 - p_challenged) * (1 - p_contessa) * _opp_influence_value(target_influence)
    # Branch 3: not challenged, target blocks Contessa (accept block)
    ev_blocked = (1 - p_challenged) * p_contessa * 0.0

    return coin_cost + ev_action_fail + ev_lands + ev_blocked


def ev_coup(target_influence: int) -> float:
    """EV of Coup: guaranteed influence removal, costs 7 coins."""
    coin_cost = -(COIN_VALUE * 7)
    kill_value = _opp_influence_value(target_influence)
    return coin_cost + kill_value


def ev_exchange(alive_opponents: list[str], players: dict) -> float:
    """
    EV of Exchange: difficult to quantify precisely.
    Approximate: mildly positive as it refreshes hand information.
    Bonus when under pressure (others have many coins).
    """
    max_opp_coins = max(
        (int(players[name].coins) for name in alive_opponents),
        default=0,
    )
    pressure_bonus = max_opp_coins / 10.0
    return 0.5 + 0.8 * pressure_bonus


def compute_all_ev(
    *,
    belief: dict,
    players: dict,
    perspective: str,
    me_probs: dict,
    alive_opponents: list[str],
    best_target: str | None,
    legal_actions: list[str],
) -> dict[str, float]:
    """
    Compute 2-ply EV for all legal actions.
    Returns a dict mapping action_title → EV float.
    """
    results = {}
    me_state = players.get(perspective)
    if me_state is None:
        return results

    my_influence = int(me_state.influence_alive)

    results["Income"] = ev_income()

    if "Foreign Aid" in legal_actions:
        results["Foreign Aid"] = ev_foreign_aid(belief, alive_opponents, my_influence)

    if "Tax" in legal_actions:
        my_duke_prob = float(me_probs.get(Role.DUKE, 0.20))
        results["Tax (Duke claim)"] = ev_tax(belief, alive_opponents, my_duke_prob, my_influence)

    if "Steal" in legal_actions and best_target:
        target_state = players[best_target]
        my_captain_prob = float(me_probs.get(Role.CAPTAIN, 0.20))
        results[f"Steal from {best_target}"] = ev_steal(
            belief=belief,
            target=best_target,
            target_coins=int(target_state.coins),
            target_influence=int(target_state.influence_alive),
            my_captain_prob=my_captain_prob,
            my_influence=my_influence,
        )

    if "Assassinate" in legal_actions and best_target:
        target_state = players[best_target]
        my_assassin_prob = float(me_probs.get(Role.ASSASSIN, 0.20))
        results[f"Assassinate {best_target}"] = ev_assassinate(
            belief=belief,
            target=best_target,
            target_influence=int(target_state.influence_alive),
            my_assassin_prob=my_assassin_prob,
            my_influence=my_influence,
        )

    if "Coup" in legal_actions and best_target:
        target_state = players[best_target]
        results[f"Coup {best_target}"] = ev_coup(int(target_state.influence_alive))

    if "Exchange" in legal_actions:
        results["Exchange (Ambassador claim)"] = ev_exchange(alive_opponents, players)

    return results
