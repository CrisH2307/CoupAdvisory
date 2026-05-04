from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

from coup.research.heuristic_baselines import benchmark_heuristic_policies


def train_behavior_clone(transitions_csv, out_dir, *, seed=42):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    transitions_csv = Path(transitions_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(transitions_csv)
    feature_cols = [col for col in frame.columns if col.startswith("s_")]
    if not feature_cols:
        raise ValueError("No state features found in transitions CSV")

    x = frame[feature_cols]
    y = frame["action_name"].astype(str)

    if len(frame) >= 8 and y.nunique() > 1:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=seed,
            stratify=y,
        )
    else:
        x_train, y_train = x, y
        x_test, y_test = x, y

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1500,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }

    with (out_dir / "bc_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "actions": sorted(y.unique()),
            },
            handle,
        )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def benchmark_policy_vs_random(transitions_csv, model_dir, *, episodes=None, seed=42):
    import numpy as np
    import pandas as pd

    del episodes

    transitions_csv = Path(transitions_csv)
    model_dir = Path(model_dir)

    frame = pd.read_csv(transitions_csv)
    with (model_dir / "bc_model.pkl").open("rb") as handle:
        payload = pickle.load(handle)

    model = payload["model"]
    feature_cols = payload["feature_cols"]
    actions = sorted(frame["action_name"].astype(str).unique())

    x = frame[feature_cols]
    y = frame["action_name"].astype(str)
    reward = frame.get("reward", pd.Series([0.0] * len(frame))).astype(float)

    pred_bc = model.predict(x)
    rng = random.Random(seed)
    pred_random = [rng.choice(actions) for _ in range(len(frame))]

    bc_match = np.array(pred_bc == y)
    random_match = np.array([pred == truth for pred, truth in zip(pred_random, y)])

    bc_step_reward = reward.to_numpy() + bc_match.astype(float)
    random_step_reward = reward.to_numpy() + random_match.astype(float)

    benchmark = {
        "behavior_cloning": {
            "action_match_rate": float(bc_match.mean()) if len(bc_match) else 0.0,
            "avg_reward_per_step": float(bc_step_reward.mean()) if len(bc_step_reward) else 0.0,
            "avg_reward_per_episode": float(bc_step_reward.mean()) if len(bc_step_reward) else 0.0,
        },
        "random_policy": {
            "action_match_rate": float(random_match.mean()) if len(random_match) else 0.0,
            "avg_reward_per_step": float(random_step_reward.mean())
            if len(random_step_reward)
            else 0.0,
            "avg_reward_per_episode": float(random_step_reward.mean())
            if len(random_step_reward)
            else 0.0,
        },
    }
    benchmark.update(benchmark_heuristic_policies(transitions_csv))

    (model_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
    return benchmark


def main():
    parser = argparse.ArgumentParser(description="Behavior cloning baseline on Markov transitions")
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train_behavior_clone(args.transitions, args.out, seed=args.seed)
    benchmark = benchmark_policy_vs_random(args.transitions, args.out, seed=args.seed)
    print(json.dumps({"train": metrics, "benchmark": benchmark}, indent=2))


if __name__ == "__main__":
    main()
