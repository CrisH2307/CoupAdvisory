from __future__ import annotations

from pathlib import Path

from coup.research.markov import ACTION_SPACE
from coup.research.rl_env import load_transition_rows

POLICY_LABELS = {
    "rule_based_agent": "Rule-Based",
    "belief_ev_agent": "Belief-EV (Non-Learning)",
    "pressure_heuristic_agent": "Pressure Heuristic",
}

HEURISTIC_POLICY_FUNCTIONS = {
    "rule_based_agent": lambda state: _rule_based_action(state),
    "belief_ev_agent": lambda state: _belief_ev_action(state),
    "pressure_heuristic_agent": lambda state: _pressure_heuristic_action(state),
}


def benchmark_heuristic_policies(transitions_csv, *, reward_mode="shaped"):
    rows = load_transition_rows(Path(transitions_csv))
    return benchmark_heuristic_policies_from_rows(rows, reward_mode=reward_mode)


def benchmark_heuristic_policies_from_rows(rows, *, reward_mode="shaped"):
    results = {}
    for policy_name, chooser in HEURISTIC_POLICY_FUNCTIONS.items():
        results[policy_name] = _score_policy(rows, chooser, reward_mode=reward_mode)
    return results


def choose_heuristic_action(policy_name, state):
    chooser = HEURISTIC_POLICY_FUNCTIONS.get(str(policy_name))
    if chooser is None:
        raise ValueError(f"Unknown heuristic policy: {policy_name}")
    return chooser(state)


def _score_policy(rows, chooser, *, reward_mode="shaped"):
    if not rows:
        return {
            "avg_reward_per_episode": 0.0,
            "avg_reward_per_step": 0.0,
            "action_match_rate": 0.0,
        }

    total_reward = 0.0
    total_steps = 0
    total_matches = 0
    episodes = set()

    for row in rows:
        episodes.add(str(row.get("meta_game_id", "game_0")))

        state = _state_from_row(row)
        chosen = chooser(state)
        if chosen not in ACTION_SPACE:
            chosen = "Income"

        truth = str(row.get("action_name", ""))
        match = int(chosen == truth)
        base_reward = float(row.get("reward", 0.0))
        if reward_mode == "shaped":
            reward = base_reward + float(match)
        else:
            actor_is_winner = int(float(row.get("actor_is_winner", 0) or 0))
            reward = base_reward + float(match * actor_is_winner)

        total_reward += reward
        total_matches += match
        total_steps += 1

    episode_count = max(1, len(episodes))
    return {
        "avg_reward_per_episode": float(total_reward / episode_count),
        "avg_reward_per_step": float(total_reward / total_steps) if total_steps else 0.0,
        "action_match_rate": float(total_matches / total_steps) if total_steps else 0.0,
    }


def _state_from_row(row):
    state = {"actor": str(row.get("actor", "")).strip().lower()}
    for key, value in row.items():
        if not str(key).startswith("s_"):
            continue
        try:
            state[str(key)[2:]] = float(value)
        except Exception:
            continue
    return state


def _role_remaining(state, role_name):
    return _clip(_safe_float(state, f"remaining_{role_name}", 3.0), 0.0, 3.0)


def _legal_actions(state):
    actor_coins = int(_safe_float(state, "actor_coins", 0.0))
    alive_players = int(_safe_float(state, "players_alive", 2.0))
    if alive_players <= 1:
        return ["Income"]
    if actor_coins >= 10:
        return ["Coup"]

    avg_opp_coins, _max_opp_coins = _opponent_coin_stats(state)
    legal = ["Income", "Foreign Aid", "Tax", "Exchange"]
    if avg_opp_coins > 0.0:
        legal.append("Steal")
    if actor_coins >= 3:
        legal.append("Assassinate")
    if actor_coins >= 7:
        legal.append("Coup")
    return legal


def _as_legal(action_name, state):
    legal = _legal_actions(state)
    if action_name in legal:
        return action_name

    fallback_order = ["Coup", "Assassinate", "Steal", "Tax", "Foreign Aid", "Exchange", "Income"]
    for candidate in fallback_order:
        if candidate in legal:
            return candidate
    return "Income"


def _opponent_coin_stats(state):
    alive_players = max(1.0, _safe_float(state, "players_alive", 2.0))
    actor_coins = _safe_float(state, "actor_coins", 0.0)
    total_coins = max(actor_coins, _safe_float(state, "total_coins", actor_coins))

    actor_key = str(state.get("actor", "")).strip().lower()
    opponent_coins = []
    for key, value in state.items():
        if not key.endswith("_coins"):
            continue
        if key in {"actor_coins", "total_coins"}:
            continue

        base = key[: -len("_coins")]
        if actor_key and base == actor_key:
            continue

        influence_key = f"{base}_influence"
        influence = _safe_float(state, influence_key, 0.0)
        if influence <= 0.0:
            continue
        opponent_coins.append(max(0.0, float(value)))

    if opponent_coins:
        avg_opp = float(sum(opponent_coins) / len(opponent_coins))
        max_opp = float(max(opponent_coins))
        return avg_opp, max_opp

    opponents = max(1.0, alive_players - 1.0)
    avg_opp = max(0.0, (total_coins - actor_coins) / opponents)
    return float(avg_opp), float(avg_opp)


def _rule_based_action(state):
    coins = _safe_float(state, "actor_coins", 0.0)
    influence = _safe_float(state, "actor_influence", 2.0)
    alive = _safe_float(state, "players_alive", 2.0)
    avg_opp, max_opp = _opponent_coin_stats(state)

    if coins >= 10.0:
        return "Coup"
    if coins >= 7.0:
        return "Coup"
    if coins >= 3.0 and alive <= 3.0:
        return "Assassinate"
    if coins >= 3.0 and max_opp >= 5.0:
        return "Assassinate"
    if avg_opp >= 1.0:
        return _as_legal("Steal", state)
    if coins <= 1.0:
        return _as_legal("Foreign Aid", state)
    if _role_remaining(state, "duke") > 0.0:
        return _as_legal("Tax", state)
    if influence <= 1.0 and _role_remaining(state, "ambassador") > 0.0:
        return _as_legal("Exchange", state)
    if coins >= 3.0:
        return _as_legal("Assassinate", state)
    return _as_legal("Income", state)


def _belief_ev_action(state):
    coins = _safe_float(state, "actor_coins", 0.0)
    influence = _safe_float(state, "actor_influence", 2.0)
    alive = max(2.0, _safe_float(state, "players_alive", 2.0))
    avg_opp, _max_opp = _opponent_coin_stats(state)
    opp_count = max(1.0, alive - 1.0)

    p_duke = _clip(_role_remaining(state, "duke") / 3.0, 0.0, 1.0)
    p_assassin = _clip(_role_remaining(state, "assassin") / 3.0, 0.0, 1.0)
    p_captain = _clip(_role_remaining(state, "captain") / 3.0, 0.0, 1.0)
    p_ambassador = _clip(_role_remaining(state, "ambassador") / 3.0, 0.0, 1.0)
    p_contessa = _clip(_role_remaining(state, "contessa") / 3.0, 0.0, 1.0)

    p_block_fa = 1.0 - (1.0 - p_duke) ** opp_count
    p_block_steal = 1.0 - (1.0 - _clip(p_captain + p_ambassador, 0.0, 1.0)) ** opp_count
    p_block_assassinate = 1.0 - (1.0 - p_contessa) ** opp_count

    p_challenge_tax = _clip(1.0 - p_duke, 0.0, 1.0)
    p_challenge_steal = _clip(1.0 - p_captain, 0.0, 1.0)
    p_challenge_assassinate = _clip(1.0 - p_assassin, 0.0, 1.0)

    scores = {
        "Income": 1.0,
        "Foreign Aid": 2.0 * (1.0 - 0.85 * p_block_fa),
        "Tax": 3.0 - 1.2 * p_challenge_tax,
        "Steal": min(2.0, max(0.0, avg_opp)) * (1.0 - 0.75 * p_block_steal)
        - 0.3 * p_challenge_steal,
        "Assassinate": (2.3 if alive <= 3.0 else 1.7) * (1.0 - 0.8 * p_block_assassinate)
        - 0.4 * p_challenge_assassinate
        - 0.2,
        "Coup": 2.6 if alive <= 3.0 else 2.2,
        "Exchange": 0.8 + (0.25 if influence <= 1.0 else 0.0) + (0.1 if alive >= 4.0 else 0.0),
    }
    return _pick_best_legal(scores, state)


def _pressure_heuristic_action(state):
    coins = _safe_float(state, "actor_coins", 0.0)
    influence = _safe_float(state, "actor_influence", 2.0)
    alive = _safe_float(state, "players_alive", 2.0)
    avg_opp, max_opp = _opponent_coin_stats(state)

    if coins >= 10.0:
        return "Coup"

    if alive <= 2.0:
        if coins >= 7.0:
            return "Coup"
        if coins >= 3.0:
            return _as_legal("Assassinate", state)
        if avg_opp > 0.5:
            return _as_legal("Steal", state)
        if _role_remaining(state, "duke") > 0.0:
            return _as_legal("Tax", state)
        return _as_legal("Income", state)

    if max_opp >= 7.0 and coins >= 7.0:
        return "Coup"
    if coins >= 3.0 and (influence <= 1.0 or max_opp >= 5.0):
        return _as_legal("Assassinate", state)
    if max_opp >= 2.0 and avg_opp >= 0.8:
        return _as_legal("Steal", state)
    if coins <= 1.0:
        return _as_legal("Foreign Aid", state)
    if _role_remaining(state, "duke") > 0.0 and coins < 7.0:
        return _as_legal("Tax", state)
    if influence <= 1.0 and _role_remaining(state, "ambassador") > 0.0:
        return _as_legal("Exchange", state)
    if coins >= 3.0:
        return _as_legal("Assassinate", state)
    return _as_legal("Income", state)


def _pick_best_legal(scores, state):
    legal = set(_legal_actions(state))
    rank = {name: idx for idx, name in enumerate(ACTION_SPACE)}

    best_action = "Income"
    best_score = float("-inf")
    for action_name in ACTION_SPACE:
        if action_name not in legal:
            continue
        score = float(scores.get(action_name, float("-inf")))
        if score > best_score:
            best_score = score
            best_action = action_name
            continue
        if score == best_score and rank[action_name] < rank[best_action]:
            best_action = action_name
    return _as_legal(best_action, state)


def _safe_float(payload, key, default=0.0):
    try:
        return float(payload.get(key, default))
    except Exception:
        return float(default)


def _clip(value, low, high):
    value = float(value)
    if value < low:
        return float(low)
    if value > high:
        return float(high)
    return float(value)
