from __future__ import annotations

import argparse
import json
from pathlib import Path


_REQUIRED_COLUMNS = {
    "game",
    "winner",
    "turns",
    "challenges",
    "challenge_accuracy",
    "advisor_claims_total",
    "advisor_recommended_challenges",
    "advisor_actual_challenges",
    "advisor_challenge_precision",
    "advisor_challenge_recall",
    "advisor_challenge_f1",
    "advisor_challenge_accuracy",
    "advisor_outcome_accuracy",
    "advisor_outcome_score",
}


def summarize_advisor_metrics(summary_csv, *, out_dir=None):
    import pandas as pd

    summary_csv = Path(summary_csv)
    frame = pd.read_csv(summary_csv)

    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out_dir = Path(out_dir) if out_dir else summary_csv.parent / "advisor_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(out_dir / "advisor_game_rows.csv", index=False)

    numeric_cols = [
        col
        for col in frame.columns
        if col not in {"game", "winner"} and frame[col].dtype.kind in {"i", "u", "f"}
    ]

    metrics = {}
    for column in numeric_cols:
        metrics[column] = {
            "mean": float(frame[column].mean()),
            "std": float(frame[column].std(ddof=0) if len(frame[column]) > 1 else 0.0),
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
        }

    metrics_rows = []
    for metric_name, values in metrics.items():
        row = {"metric": metric_name}
        row.update(values)
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "advisor_metrics_summary.csv", index=False)

    winner_distribution = {
        str(key): int(value) for key, value in frame["winner"].value_counts().to_dict().items()
    }

    report = {
        "games": int(len(frame)),
        "metrics": metrics,
        "winner_distribution": winner_distribution,
    }
    (out_dir / "advisor_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Aggregate advisor metrics from simulation summary")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = summarize_advisor_metrics(args.summary, out_dir=args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
