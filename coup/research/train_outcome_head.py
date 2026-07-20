from __future__ import annotations

"""
Train an outcome-labeled challenge head model.

Reads a directory of game JSON files, labels each challenge opportunity
using outcome_labels.py, and trains a gradient-boosted classifier.

Usage:
    python -m coup.research.train_outcome_head \
        --sim-dir data/sim_games/ \
        --out runs/outcome_head/ \
        --target label_optimal_challenge

The output is a pickle file compatible with coup/advisor_ml.py.
"""

import argparse
import json
import pickle
from pathlib import Path

from coup.research.outcome_labels import label_challenge_opportunities


def build_dataset(sim_dir: Path, target: str) -> tuple[list[list[float]], list[int], list[str]]:
    """
    Scan sim_dir for game_*.json files, label opportunities, return X, y, feature_names.
    """
    all_rows = []
    for game_path in sorted(sim_dir.glob("game_*.json")):
        data = json.loads(game_path.read_text())
        if isinstance(data, dict):
            raw_events = data.get("events", [])
            winner = data.get("winner")
        else:
            raw_events = data
            winner = None
        rows = label_challenge_opportunities(raw_events, winner)
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError(f"No labeled rows found in {sim_dir}")

    feature_cols = [
        "f_belief_p_role",
        "f_players_alive",
        "f_claimant_coins",
        "remaining_copies",
    ]

    X = [[float(row.get(col, 0.0)) for col in feature_cols] for row in all_rows]
    y = [int(row.get(target, 0)) for row in all_rows]
    return X, y, feature_cols


def train(sim_dir: Path, out_dir: Path, target: str) -> None:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required: pip install scikit-learn") from exc

    print(f"Building dataset from {sim_dir} with target={target}...")
    X, y, feature_cols = build_dataset(sim_dir, target)
    print(f"  {len(X)} samples, {sum(y)} positive labels ({100*sum(y)/len(y):.1f}%)")

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"  CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    model.fit(X, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_cols": feature_cols,
        "target": target,
        "version": f"outcome-head-v1-{target}",
        "n_samples": len(X),
        "cv_auc_mean": float(scores.mean()),
        "cv_auc_std": float(scores.std()),
    }
    out_path = out_dir / "model.pkl"
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"  Saved model to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train outcome-labeled challenge head")
    parser.add_argument("--sim-dir", required=True, help="Directory of game_*.json files")
    parser.add_argument("--out", required=True, help="Output directory for model.pkl")
    parser.add_argument(
        "--target",
        choices=["label_optimal_challenge", "label_outcome_positive", "was_challenged"],
        default="label_optimal_challenge",
        help="Training target column",
    )
    args = parser.parse_args()
    train(Path(args.sim_dir), Path(args.out), args.target)


if __name__ == "__main__":
    main()
