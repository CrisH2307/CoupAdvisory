from __future__ import annotations

import argparse
from pathlib import Path


def generate_plots(report_dir, out_dir):
    import pandas as pd

    report_dir = Path(report_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    created = []

    metrics_path = report_dir / "metrics_table.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if not metrics.empty and "target" in metrics.columns and "accuracy" in metrics.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(metrics["target"], metrics["accuracy"])
            ax.set_title("Model Accuracy by Target")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            path = out_dir / "accuracy_by_target.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(path)

        if not metrics.empty and "target" in metrics.columns and "f1_macro" in metrics.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(metrics["target"], metrics["f1_macro"])
            ax.set_title("Macro F1 by Target")
            ax.set_ylabel("F1 Macro")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            path = out_dir / "f1_by_target.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(path)

    return created


def main():
    parser = argparse.ArgumentParser(description="Generate report plots")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = generate_plots(args.report_dir, args.out)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
