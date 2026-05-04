from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def generate_report(dataset_path, models_dir, out_dir, *, calibration_bins=10):
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    dataset_path = Path(dataset_path)
    models_dir = Path(models_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset_path)
    x_full = _feature_matrix(frame)

    targets = [
        "y_next_action",
        "y_next_is_challenge",
        "y_coup_within_horizon",
        "y_current_claim_challenged",
    ]

    metrics_rows = []
    report = {}

    for target in targets:
        model_path = models_dir / f"{target}.pkl"
        if not model_path.exists() or target not in frame.columns:
            continue

        with model_path.open("rb") as handle:
            payload = pickle.load(handle)

        model = payload["model"]
        feature_cols = payload.get("feature_cols", [])

        subset = frame.copy()
        x_subset = x_full.copy()
        if target == "y_current_claim_challenged":
            mask = subset[target] >= 0
            subset = subset[mask]
            x_subset = x_subset.loc[mask]

        if subset.empty:
            continue

        x = _align_features(x_subset, feature_cols)
        y = subset[target]
        pred = model.predict(x)

        row = {
            "target": target,
            "accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        }

        if len(set(y)) == 2 and hasattr(model, "predict_proba"):
            try:
                score = model.predict_proba(x)[:, 1]
                row["roc_auc"] = float(roc_auc_score(y, score))
                _write_calibration_csv(
                    out_dir / f"calibration_{target}.csv",
                    y,
                    score,
                    bins=int(calibration_bins),
                )
            except Exception:
                pass

        report[target] = row
        metrics_rows.append(row)

        confusion = pd.crosstab(
            pd.Series(y, name="actual"),
            pd.Series(pred, name="predicted"),
            dropna=False,
        )
        confusion.to_csv(out_dir / f"confusion_{target}.csv")

    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_table.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _write_calibration_csv(path, y_true, y_score, *, bins):
    import pandas as pd

    bins = max(1, int(bins))
    frame = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    frame["bin"] = pd.cut(frame["y_score"], bins=bins, include_lowest=True)
    grouped = frame.groupby("bin", observed=True).agg(
        count=("y_true", "size"),
        avg_score=("y_score", "mean"),
        positive_rate=("y_true", "mean"),
    )
    grouped.reset_index().to_csv(path, index=False)


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

    return pd.concat([numeric, categorical], axis=1).fillna(0)


def _align_features(frame, feature_cols):
    import pandas as pd

    aligned = pd.DataFrame(index=frame.index)
    for col in feature_cols:
        if col in frame.columns:
            aligned[col] = frame[col]
        else:
            aligned[col] = 0.0
    return aligned


def main():
    parser = argparse.ArgumentParser(description="Generate research report artifacts")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    report = generate_report(args.dataset, args.models, args.out, calibration_bins=args.bins)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
