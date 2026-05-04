from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..schemas import MetricsResponse, MetricsSnapshot

router = APIRouter(prefix="/api/research", tags=["research"])

_RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs"


def _load_advisor_report(path: Path) -> dict | None:
    report_path = path / "advisor_eval" / "advisor_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text())


def _policy_row(name: str, run_dir: Path) -> MetricsSnapshot | None:
    report = _load_advisor_report(run_dir)
    if report is None:
        return None
    metrics = report.get("metrics", {})

    def _mean(key: str) -> float | None:
        entry = metrics.get(key)
        if entry is None:
            return None
        return float(entry.get("mean"))

    return MetricsSnapshot(
        policy=name,
        games=int(report.get("games", 0)),
        precision=_mean("advisor_challenge_precision"),
        recall=_mean("advisor_challenge_recall"),
        f1=_mean("advisor_challenge_f1"),
        accuracy=_mean("advisor_challenge_accuracy"),
        outcome_accuracy=_mean("advisor_outcome_accuracy"),
    )


def _load_ml_row(name: str, path: Path) -> MetricsSnapshot | None:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text())
    return MetricsSnapshot(
        policy=name,
        games=0,
        accuracy=float(data.get("accuracy")) if "accuracy" in data else None,
        f1=float(data.get("f1_macro")) if "f1_macro" in data else None,
        extra=data,
    )


def _load_dqn_row(path: Path) -> MetricsSnapshot | None:
    bench_path = path / "benchmark.json"
    if not bench_path.exists():
        return None
    data = json.loads(bench_path.read_text())
    dqn = data.get("dqn_policy", {})
    return MetricsSnapshot(
        policy="dqn",
        games=0,
        accuracy=float(dqn.get("action_match_rate")) if "action_match_rate" in dqn else None,
        extra=data,
    )


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(run_root: str | None = None) -> MetricsResponse:
    root = Path(run_root) if run_root else _RUNS_ROOT
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"runs dir not found: {root}")

    rows: list[MetricsSnapshot] = []
    pairs = [
        ("heuristic", root / "bench_heuristic"),
        ("never_challenge", root / "bench_never"),
        ("always_challenge", root / "bench_always"),
    ]
    for name, path in pairs:
        row = _policy_row(name, path)
        if row is not None:
            rows.append(row)

    bc_row = _load_ml_row("behavior_cloning", root / "bench_heuristic" / "bc")
    if bc_row is not None:
        rows.append(bc_row)

    dqn_row = _load_dqn_row(root / "bench_heuristic" / "rl")
    if dqn_row is not None:
        rows.append(dqn_row)

    return MetricsResponse(policies=rows)


@router.get("/runs")
def list_runs(run_root: str | None = None) -> dict:
    root = Path(run_root) if run_root else _RUNS_ROOT
    if not root.exists():
        return {"runs": []}
    return {"runs": sorted(p.name for p in root.iterdir() if p.is_dir())}
