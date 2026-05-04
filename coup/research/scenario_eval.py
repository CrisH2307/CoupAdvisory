from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

from coup.research.heuristic_baselines import POLICY_LABELS, choose_heuristic_action
from coup.research.markov import ACTION_SPACE
from coup.research.rl_env import load_transition_rows
from coup.research.rl_train import load_dqn_policy


def evaluate_policies_by_scenario(
    *,
    transitions_csv,
    bc_model_dir=None,
    dqn_model_path=None,
    seed=42,
    max_rows_per_scenario=None,
):
    rows = load_transition_rows(Path(transitions_csv))
    if not rows:
        return []

    scenario_specs = _scenario_specs()
    policy_choosers = _build_policy_choosers(
        rows=rows,
        bc_model_dir=bc_model_dir,
        dqn_model_path=dqn_model_path,
        seed=seed,
    )
    return _score_by_scenario(
        rows=rows,
        scenario_specs=scenario_specs,
        policy_choosers=policy_choosers,
        max_rows_per_scenario=max_rows_per_scenario,
    )


def write_scenario_eval_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_policy_choosers(*, rows, bc_model_dir, dqn_model_path, seed):
    choosers = {
        "random_policy": _random_chooser(seed=seed),
        "rule_based_agent": lambda row, state, idx: choose_heuristic_action("rule_based_agent", state),
        "belief_ev_agent": lambda row, state, idx: choose_heuristic_action("belief_ev_agent", state),
        "pressure_heuristic_agent": lambda row, state, idx: choose_heuristic_action(
            "pressure_heuristic_agent", state
        ),
    }

    bc_chooser = _bc_chooser(bc_model_dir, rows=rows)
    if bc_chooser is not None:
        choosers["behavior_cloning"] = bc_chooser

    dqn_chooser = _dqn_chooser(rows=rows, dqn_model_path=dqn_model_path)
    if dqn_chooser is not None:
        choosers["dqn_policy"] = dqn_chooser

    return choosers


def _bc_chooser(bc_model_dir, *, rows):
    if not bc_model_dir:
        return None
    model_path = Path(bc_model_dir) / "bc_model.pkl"
    if not model_path.exists():
        return None

    try:
        import pandas as pd
    except Exception:
        return None

    with model_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        model = payload.get("model")
        feature_cols = list(payload.get("feature_cols", []))
    else:
        # Backward compatibility: older files may store the sklearn model directly.
        model = payload
        feature_cols = []

    if model is None or not hasattr(model, "predict"):
        return None

    if not feature_cols:
        feature_cols = _transition_feature_cols(rows)
    if not feature_cols:
        return None

    def choose(row, state, index):
        del state, index
        features = {name: float(row.get(name, 0.0)) for name in feature_cols}
        frame = pd.DataFrame([features], columns=feature_cols)
        try:
            prediction = model.predict(frame)[0]
        except Exception:
            return "Income"
        return str(prediction)

    return choose


def _dqn_chooser(*, rows, dqn_model_path):
    if not dqn_model_path:
        return None
    model_file = Path(dqn_model_path)
    if not model_file.exists():
        return None

    model = load_dqn_policy(model_file)
    if model is None:
        return None

    state_keys = sorted(
        key
        for key in rows[0].keys()
        if key.startswith("s_") and not key.startswith("s_ns_")
    )
    if not state_keys:
        return None

    labels = list(getattr(model, "action_labels", [])) or list(ACTION_SPACE)

    def choose(row, state, index):
        del state, index
        vector = [float(row.get(name, 0.0)) for name in state_keys]
        action_index = int(model.act(vector))
        if action_index < 0 or action_index >= len(labels):
            return "Income"
        return str(labels[action_index])

    return choose


def _random_chooser(*, seed):
    rng = random.Random(int(seed))

    def choose(row, state, index):
        del row, state, index
        return rng.choice(ACTION_SPACE)

    return choose


def _scenario_specs():
    return [
        {
            "name": "reference_all_turns",
            "preferred_actions": None,
            "predicate": lambda state: True,
        },
        {
            "name": "forced_coup_state",
            "preferred_actions": {"Coup"},
            "predicate": lambda state: _safe_float(state, "actor_coins", 0.0) >= 10.0,
        },
        {
            "name": "coup_ready_state",
            "preferred_actions": {"Coup", "Assassinate", "Steal"},
            "predicate": lambda state: 7.0 <= _safe_float(state, "actor_coins", 0.0) < 10.0,
        },
        {
            "name": "assassinate_window",
            "preferred_actions": {"Assassinate", "Steal", "Tax"},
            "predicate": lambda state: (
                3.0 <= _safe_float(state, "actor_coins", 0.0) < 7.0
                and _safe_float(state, "players_alive", 2.0) <= 3.0
            ),
        },
        {
            "name": "steal_opportunity",
            "preferred_actions": {"Steal", "Tax", "Income"},
            "predicate": lambda state: (
                _safe_float(state, "actor_coins", 0.0) < 10.0
                and _opponent_max_coins(state) >= 2.0
            ),
        },
        {
            "name": "tax_pressure_window",
            "preferred_actions": {"Tax", "Foreign Aid", "Income"},
            "predicate": lambda state: (
                _safe_float(state, "actor_coins", 0.0) <= 6.0
                and _safe_float(state, "remaining_duke", 0.0) > 0.0
                and _safe_float(state, "players_alive", 2.0) >= 3.0
            ),
        },
        {
            "name": "endgame_one_influence",
            "preferred_actions": {"Coup", "Assassinate", "Steal", "Income"},
            "predicate": lambda state: (
                int(_safe_float(state, "players_alive", 2.0)) == 2
                and _safe_float(state, "actor_influence", 2.0) <= 1.0
            ),
        },
    ]


def _score_by_scenario(
    *,
    rows,
    scenario_specs,
    policy_choosers,
    max_rows_per_scenario=None,
):
    aggregates = {}
    seen_counts = {spec["name"]: 0 for spec in scenario_specs}

    for index, row in enumerate(rows):
        state = _state_from_row(row)
        legal_actions = set(_legal_actions(state))
        logged_action = str(row.get("action_name", ""))
        base_reward = float(row.get("reward", 0.0))

        matching_specs = [spec for spec in scenario_specs if bool(spec["predicate"](state))]
        if not matching_specs:
            continue

        if max_rows_per_scenario is not None:
            filtered = []
            limit = int(max_rows_per_scenario)
            for spec in matching_specs:
                if seen_counts[spec["name"]] >= limit:
                    continue
                seen_counts[spec["name"]] += 1
                filtered.append(spec)
            matching_specs = filtered
            if not matching_specs:
                continue

        for policy_name, chooser in policy_choosers.items():
            chosen_action = str(chooser(row, state, index))
            if chosen_action not in ACTION_SPACE:
                chosen_action = "Income"

            match = int(chosen_action == logged_action)
            reward = base_reward + float(match)
            legal = int(chosen_action in legal_actions)

            for spec in matching_specs:
                key = (spec["name"], policy_name)
                bucket = aggregates.setdefault(
                    key,
                    {
                        "cases": 0,
                        "match_sum": 0.0,
                        "reward_sum": 0.0,
                        "legal_sum": 0.0,
                        "preferred_sum": 0.0,
                    },
                )
                bucket["cases"] += 1
                bucket["match_sum"] += float(match)
                bucket["reward_sum"] += float(reward)
                bucket["legal_sum"] += float(legal)

                preferred_actions = spec.get("preferred_actions")
                if preferred_actions is None:
                    preferred = 1
                else:
                    preferred = int(chosen_action in preferred_actions)
                bucket["preferred_sum"] += float(preferred)

    rows_out = []
    for (scenario_name, policy_name), bucket in sorted(aggregates.items()):
        cases = int(bucket["cases"])
        if cases <= 0:
            continue

        policy_label = POLICY_LABELS.get(policy_name)
        if policy_name == "random_policy":
            policy_label = "Random"
        if policy_name == "behavior_cloning":
            policy_label = "Behavior Cloning"
        if policy_name == "dqn_policy":
            policy_label = "DQN"
        if policy_label is None:
            policy_label = policy_name

        rows_out.append(
            {
                "scenario": scenario_name,
                "policy_key": policy_name,
                "policy": policy_label,
                "cases": cases,
                "action_match_rate": float(bucket["match_sum"] / cases),
                "avg_reward_per_step": float(bucket["reward_sum"] / cases),
                "legal_rate": float(bucket["legal_sum"] / cases),
                "preferred_action_rate": float(bucket["preferred_sum"] / cases),
            }
        )
    return rows_out


def _state_from_row(row):
    state = {"actor": str(row.get("actor", "")).strip().lower()}
    for key, value in row.items():
        if not str(key).startswith("s_"):
            continue
        name = str(key)[2:]
        try:
            state[name] = float(value)
        except Exception:
            continue
    return state


def _transition_feature_cols(rows):
    if not rows:
        return []
    first = rows[0]
    cols = []
    for key in first.keys():
        key = str(key)
        if not key.startswith("s_"):
            continue
        if key.startswith("s_ns_"):
            continue
        cols.append(key)
    return cols


def _legal_actions(state):
    coins = _safe_float(state, "actor_coins", 0.0)
    players_alive = int(_safe_float(state, "players_alive", 2.0))

    if players_alive <= 1:
        return ["Income"]
    if coins >= 10.0:
        return ["Coup"]

    actions = ["Income", "Foreign Aid", "Tax", "Steal", "Exchange"]
    if coins >= 3.0:
        actions.append("Assassinate")
    if coins >= 7.0:
        actions.append("Coup")
    return actions


def _opponent_max_coins(state):
    actor_key = str(state.get("actor", "")).strip().lower()
    values = []
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
        values.append(max(0.0, float(value)))
    if values:
        return float(max(values))
    return 0.0


def _safe_float(payload, key, default=0.0):
    try:
        return float(payload.get(key, default))
    except Exception:
        return float(default)


def _parse_args():
    parser = argparse.ArgumentParser(description="Scenario-based evaluation for Coup policies")
    parser.add_argument("--transitions", required=True, help="Path to transitions CSV")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--bc-model-dir", default=None, help="Optional BC model directory")
    parser.add_argument("--dqn-model-path", default=None, help="Optional DQN model path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-scenario", type=int, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    rows = evaluate_policies_by_scenario(
        transitions_csv=args.transitions,
        bc_model_dir=args.bc_model_dir,
        dqn_model_path=args.dqn_model_path,
        seed=args.seed,
        max_rows_per_scenario=args.max_rows_per_scenario,
    )
    write_scenario_eval_csv(rows, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
