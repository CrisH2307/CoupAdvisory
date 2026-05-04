from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from coup.advisor_ml import (
    ACTION_OPTIONS,
    BLOCK_OPTIONS,
    FEATURE_COLUMNS,
    ROLE_ORDER,
)

MODEL_VERSION = "ch-head-v1"


def _action_key(name: str) -> str:
    return f"action_{name.lower().replace(' ', '_')}"


def _block_key(name: str) -> str:
    return f"block_{name.lower().replace(' ', '_')}"


def _role_key(value: str) -> str:
    return f"role_{value.lower()}"


def _frame_to_features(frame):
    import numpy as np
    import pandas as pd

    base = pd.DataFrame(0.0, index=frame.index, columns=FEATURE_COLUMNS)

    numeric_map = {
        "f_players_alive": "f_players_alive",
        "f_total_coins": "f_total_coins",
        "f_actor_coins": "f_actor_coins",
        "f_actor_influence": "f_actor_influence",
        "f_remaining_duke": "f_remaining_duke",
        "f_remaining_assassin": "f_remaining_assassin",
        "f_remaining_captain": "f_remaining_captain",
        "f_remaining_ambassador": "f_remaining_ambassador",
        "f_remaining_contessa": "f_remaining_contessa",
        "f_belief_p_truth": "f_belief_p_truth",
    }
    for dst, src in numeric_map.items():
        if src in frame.columns:
            base[dst] = frame[src].astype(float).to_numpy()

    remaining_claimed = np.zeros(len(frame), dtype=float)
    role_values = frame["current_claimed_role"].astype(str).to_numpy()
    for role in ROLE_ORDER:
        mask = role_values == role.value
        key = _role_key(role.value)
        base.loc[mask, key] = 1.0
        remaining_col = f"f_remaining_{role.value.lower()}"
        if remaining_col in frame.columns:
            remaining_claimed[mask] = frame.loc[mask, remaining_col].astype(float).to_numpy()
    base["f_remaining_claimed"] = remaining_claimed

    actions = frame.get("current_action_name", pd.Series([""] * len(frame))).astype(str)
    for action in ACTION_OPTIONS:
        mask = actions.to_numpy() == action
        base.loc[mask, _action_key(action)] = 1.0

    blocks = frame.get("current_block_type", pd.Series([""] * len(frame))).astype(str)
    for block in BLOCK_OPTIONS:
        mask = blocks.to_numpy() == block
        base.loc[mask, _block_key(block)] = 1.0

    return base


def train_challenge_head(dataset_csv, out_dir, *, seed=42, test_size=0.25):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset_csv = Path(dataset_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset_csv)
    frame = frame[frame["y_current_claim_challenged"].isin([0, 1])].copy()
    frame = frame[frame["current_claimed_role"].astype(str) != ""].copy()
    if len(frame) < 20:
        raise ValueError(f"Too few labeled claim rows ({len(frame)}) to train a model.")

    x = _frame_to_features(frame)
    y = frame["y_current_claim_challenged"].astype(int).to_numpy()

    stratify = y if len(set(y)) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    def _score(name, xs, ys):
        pred = pipeline.predict(xs)
        return {
            f"{name}_accuracy": float(accuracy_score(ys, pred)),
            f"{name}_precision": float(precision_score(ys, pred, zero_division=0)),
            f"{name}_recall": float(recall_score(ys, pred, zero_division=0)),
            f"{name}_f1": float(f1_score(ys, pred, zero_division=0)),
        }

    metrics: dict = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate": float(y.mean()),
        "version": MODEL_VERSION,
    }
    metrics.update(_score("train", x_train, y_train))
    metrics.update(_score("test", x_test, y_test))

    heuristic_pred = (frame["f_belief_p_truth"].astype(float) < 0.25).astype(int).to_numpy()
    metrics["heuristic_baseline_f1"] = float(
        f1_score(y, heuristic_pred, zero_division=0)
    )
    metrics["heuristic_baseline_accuracy"] = float(accuracy_score(y, heuristic_pred))

    model_path = out_dir / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": pipeline,
                "feature_cols": list(FEATURE_COLUMNS),
                "version": MODEL_VERSION,
            },
            handle,
        )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the challenge-head classifier")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    metrics = train_challenge_head(
        args.dataset, args.out, seed=args.seed, test_size=args.test_size
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
