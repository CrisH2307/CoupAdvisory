from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def train_baseline(dataset_path, out_dir, *, seed=42):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset_path)

    tasks = [
        "y_next_action",
        "y_next_is_challenge",
        "y_coup_within_horizon",
        "y_current_claim_challenged",
    ]

    reports = {}
    for target in tasks:
        if target not in frame.columns:
            continue

        subset = frame.copy()
        if target == "y_current_claim_challenged":
            subset = subset[subset[target] >= 0]
            if subset.empty:
                reports[target] = {
                    "accuracy": 0.0,
                    "f1_macro": 0.0,
                }
                continue

        x_sub = _feature_matrix(subset)
        y_sub = subset[target]

        if y_sub.nunique() < 2:
            model = RandomForestClassifier(n_estimators=50, random_state=seed)
            model.fit(x_sub, y_sub)
            pred = model.predict(x_sub)
            report = {
                "accuracy": float(accuracy_score(y_sub, pred)),
                "f1_macro": float(f1_score(y_sub, pred, average="macro", zero_division=0)),
            }
            _save_model(out_dir, target, model, list(x_sub.columns), task_type="single_class")
            reports[target] = report
            continue

        x_train, x_test, y_train, y_test = train_test_split(
            x_sub,
            y_sub,
            test_size=0.25,
            random_state=seed,
            stratify=y_sub,
        )

        model = RandomForestClassifier(n_estimators=250, random_state=seed)
        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        report = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        }

        if y_sub.nunique() == 2:
            try:
                y_score = model.predict_proba(x_test)[:, 1]
                report["roc_auc"] = float(roc_auc_score(y_test, y_score))
            except Exception:
                pass

        _save_model(
            out_dir,
            target,
            model,
            list(x_sub.columns),
            task_type="multiclass" if y_sub.nunique() > 2 else "binary",
        )
        reports[target] = report

    (out_dir / "metrics.json").write_text(json.dumps(reports, indent=2))
    return reports


def _feature_matrix(frame):
    import pandas as pd

    numeric_cols = [col for col in frame.columns if col.startswith("f_")]
    numeric_cols += [col for col in ["meta_step"] if col in frame.columns]

    categorical_cols = [
        col
        for col in [
            "current_event_type",
            "current_action_name",
            "current_actor",
            "current_target",
        ]
        if col in frame.columns
    ]

    numeric = frame[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=frame.index)
    categorical = (
        pd.get_dummies(frame[categorical_cols].fillna(""), prefix=categorical_cols)
        if categorical_cols
        else pd.DataFrame(index=frame.index)
    )

    x = pd.concat([numeric, categorical], axis=1)
    return x.fillna(0)


def _save_model(out_dir, target, model, feature_cols, *, task_type):
    payload = {
        "model": model,
        "feature_cols": feature_cols,
        "task_type": task_type,
    }
    with (Path(out_dir) / f"{target}.pkl").open("wb") as handle:
        pickle.dump(payload, handle)


def main():
    parser = argparse.ArgumentParser(description="Train baseline research models")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = train_baseline(args.dataset, args.out, seed=args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
