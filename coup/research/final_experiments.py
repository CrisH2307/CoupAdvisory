from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

from coup.research.advisor_eval import summarize_advisor_metrics
from coup.research.bc import benchmark_policy_vs_random, train_behavior_clone
from coup.research.heuristic_baselines import POLICY_LABELS
from coup.research.markov import build_transition_dataset_from_sim_dir, write_transition_csv
from coup.research.rl_train import train_dqn_policy
from coup.research.scenario_eval import evaluate_policies_by_scenario, write_scenario_eval_csv
from coup.sim.metrics import summarize_game
from coup.sim.sim import simulate_games

DEFAULT_SEEDS = (101, 202, 303, 404, 505)
DEFAULT_CONDITIONS = (
    {
        "name": "aggressive-heavy",
        "bot_mix": ["aggressive", "aggressive", "bluffer", "aggressive"],
        "seed_offset": 0,
    },
    {
        "name": "honest-heavy",
        "bot_mix": ["honest", "honest", "cautious", "honest"],
        "seed_offset": 100_000,
    },
)


def run_final_experiments(out_root, *, seeds=None, games_per_condition=50, players=4, max_turns=200, dqn_episodes=100, dqn_seed=42, skip_figures=False, figure_fallback_dir=None):  # noqa: E501
    pd, _np = _import_tabular()
    run_root = _prepare_run_root(Path(out_root))

    decision_dir = run_root / "decision"
    decision_dir.mkdir(parents=True, exist_ok=True)
    sim_all_dir = run_root / "sim_all"
    sim_all_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = run_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed_values = list(seeds or DEFAULT_SEEDS)
    all_rows = []
    game_counter = 0

    for condition in DEFAULT_CONDITIONS:
        condition_rows = []
        condition_name = condition["name"]
        condition_mix = condition["bot_mix"]
        condition_root = decision_dir / condition_name
        condition_root.mkdir(parents=True, exist_ok=True)

        for seed in seed_values:
            seed_root = condition_root / f"seed_{seed}"
            seed_root.mkdir(parents=True, exist_ok=True)
            game_seed = int(seed) + int(condition["seed_offset"])

            results = simulate_games(
                int(games_per_condition),
                players=int(players),
                seed=game_seed,
                bot_mix=condition_mix,
                max_turns=int(max_turns),
            )

            seed_summary_rows = []
            for game_index, result in enumerate(results, start=1):
                metrics_standard = summarize_game(
                    events=result.events,
                    players=result.players,
                    winner=result.winner,
                    turns=result.turns,
                    strict_no_duplicate_hand=False,
                    advisor_policy="advisor",
                )
                metrics_strict = summarize_game(
                    events=result.events,
                    players=result.players,
                    winner=result.winner,
                    turns=result.turns,
                    strict_no_duplicate_hand=True,
                    advisor_policy="advisor",
                )
                metrics_off = summarize_game(
                    events=result.events,
                    players=result.players,
                    winner=result.winner,
                    turns=result.turns,
                    strict_no_duplicate_hand=False,
                    advisor_policy="never_challenge",
                )

                seed_summary_rows.append(
                    {
                        "game": game_index,
                        "winner": metrics_standard.winner,
                        "turns": metrics_standard.turns,
                        "challenges": metrics_standard.challenges,
                        "challenge_accuracy": metrics_standard.challenge_accuracy,
                        "advisor_agreement_rate": metrics_standard.advisor_agreement_rate,
                        "advisor_claims_total": metrics_standard.advisor_claims_total,
                        "advisor_recommended_challenges": (
                            metrics_standard.advisor_recommended_challenges
                        ),
                        "advisor_actual_challenges": metrics_standard.advisor_actual_challenges,
                        "advisor_challenge_precision": metrics_standard.advisor_challenge_precision,
                        "advisor_challenge_recall": metrics_standard.advisor_challenge_recall,
                        "advisor_challenge_f1": metrics_standard.advisor_challenge_f1,
                        "advisor_challenge_accuracy": metrics_standard.advisor_challenge_accuracy,
                        "advisor_outcome_accuracy": metrics_standard.advisor_outcome_accuracy,
                        "advisor_outcome_score": metrics_standard.advisor_outcome_score,
                    }
                )

                game_counter += 1
                game_path = sim_all_dir / f"game_{game_counter}.json"
                game_path.write_text(json.dumps(result.events, indent=2))

                row = {
                    "condition": condition_name,
                    "bot_mix": ",".join(condition_mix),
                    "seed": int(seed),
                    "game": int(game_index),
                    "winner": result.winner,
                    "turns": result.turns,
                    "standard_challenge_f1": metrics_standard.advisor_challenge_f1,
                    "standard_outcome_accuracy": metrics_standard.advisor_outcome_accuracy,
                    "standard_outcome_score": metrics_standard.advisor_outcome_score,
                    "strict_challenge_f1": metrics_strict.advisor_challenge_f1,
                    "strict_outcome_accuracy": metrics_strict.advisor_outcome_accuracy,
                    "strict_outcome_score": metrics_strict.advisor_outcome_score,
                    "off_challenge_f1": metrics_off.advisor_challenge_f1,
                    "off_outcome_accuracy": metrics_off.advisor_outcome_accuracy,
                    "off_outcome_score": metrics_off.advisor_outcome_score,
                    "off_recommended_challenges": metrics_off.advisor_recommended_challenges,
                }
                all_rows.append(row)
                condition_rows.append(row)

            summary_csv = seed_root / "summary.csv"
            _write_csv(
                summary_csv,
                fieldnames=list(seed_summary_rows[0].keys()) if seed_summary_rows else [],
                rows=seed_summary_rows,
            )
            summarize_advisor_metrics(summary_csv, out_dir=seed_root / "advisor_eval")

        condition_frame = pd.DataFrame(
            [row for row in condition_rows if row["condition"] == condition_name]
        )
        if not condition_frame.empty:
            condition_frame.to_csv(condition_root / "all_game_rows.csv", index=False)

    all_frame = pd.DataFrame(all_rows)
    all_rows_path = tables_dir / "all_game_rows.csv"
    all_frame.to_csv(all_rows_path, index=False)

    tables = _build_tables(all_frame, tables_dir)
    transitions_path = run_root / "markov_transitions.csv"
    transition_rows = build_transition_dataset_from_sim_dir(sim_all_dir)
    write_transition_csv(transition_rows, transitions_path)

    bc_dir = run_root / "bc_model"
    bc_train_metrics = train_behavior_clone(transitions_path, bc_dir, seed=dqn_seed)
    bc_benchmark = benchmark_policy_vs_random(transitions_path, bc_dir, seed=dqn_seed)

    dqn_dir = run_root / "dqn_model"
    dqn_summary = train_dqn_policy(
        transitions_path,
        dqn_dir,
        episodes=int(dqn_episodes),
        seed=int(dqn_seed),
        compare_bc_dir=bc_dir,
    )

    policy_table, training_table = _build_policy_tables(
        pd,
        bc_train_metrics,
        bc_benchmark,
        dqn_summary,
        dqn_dir,
    )
    policy_table_path = tables_dir / "table_policy_benchmark.csv"
    training_table_path = tables_dir / "table_training_summary.csv"
    policy_table.to_csv(policy_table_path, index=False)
    training_table.to_csv(training_table_path, index=False)

    scenario_table_name = None
    try:
        scenario_rows = evaluate_policies_by_scenario(
            transitions_csv=transitions_path,
            bc_model_dir=bc_dir,
            dqn_model_path=dqn_dir / "dqn_model.pt",
            seed=int(dqn_seed),
        )
        if scenario_rows:
            scenario_table_name = "table_scenario_policy_eval.csv"
            write_scenario_eval_csv(scenario_rows, tables_dir / scenario_table_name)
    except Exception as exc:
        (tables_dir / "scenario_eval_error.txt").write_text(str(exc))

    if skip_figures:
        figure_manifest = _copy_fallback_figures(
            figures_dir,
            fallback_dir=figure_fallback_dir,
        )
    else:
        figure_manifest = _build_figures(
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            dqn_dir=dqn_dir,
            fallback_dir=figure_fallback_dir,
        )

    manifest = {
        "run_root": str(run_root),
        "config": {
            "seeds": seed_values,
            "conditions": DEFAULT_CONDITIONS,
            "games_per_condition": int(games_per_condition),
            "players": int(players),
            "max_turns": int(max_turns),
            "dqn_episodes": int(dqn_episodes),
            "dqn_seed": int(dqn_seed),
            "skip_figures": bool(skip_figures),
        },
        "tables": tables
        + [policy_table_path.name, training_table_path.name]
        + ([scenario_table_name] if scenario_table_name else []),
        "figures": figure_manifest,
        "paths": {
            "tables_dir": str(tables_dir),
            "figures_dir": str(figures_dir),
            "transitions_csv": str(transitions_path),
            "all_rows_csv": str(all_rows_path),
        },
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _prepare_run_root(out_root):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = out_root / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _build_tables(all_frame, tables_dir):
    pd, _np = _import_tabular()
    group_cols = ["seed", "condition", "bot_mix"]
    seed_condition = (
        all_frame.groupby(group_cols, as_index=False)
        .agg(
            standard_challenge_f1=("standard_challenge_f1", "mean"),
            standard_outcome_accuracy=("standard_outcome_accuracy", "mean"),
            standard_outcome_score=("standard_outcome_score", "mean"),
            strict_challenge_f1=("strict_challenge_f1", "mean"),
            strict_outcome_accuracy=("strict_outcome_accuracy", "mean"),
            strict_outcome_score=("strict_outcome_score", "mean"),
            off_challenge_f1=("off_challenge_f1", "mean"),
            off_outcome_accuracy=("off_outcome_accuracy", "mean"),
            off_outcome_score=("off_outcome_score", "mean"),
        )
    )
    seed_condition.to_csv(tables_dir / "seed_condition_metrics.csv", index=False)

    main_rows = []
    for metric_name, column in (
        ("Challenge F1", "standard_challenge_f1"),
        ("Outcome Accuracy", "standard_outcome_accuracy"),
        ("Outcome Score", "standard_outcome_score"),
    ):
        summary = _summary_stats(seed_condition[column].tolist())
        main_rows.append(
            {
                "metric": metric_name,
                "mean": summary["mean"],
                "std": summary["std"],
                "ci95_low": summary["ci95_low"],
                "ci95_high": summary["ci95_high"],
            }
        )
    table_main = pd.DataFrame(main_rows)
    table_main.to_csv(tables_dir / "table_advisor_main.csv", index=False)

    botmix_rows = []
    for condition_name, frame in seed_condition.groupby("condition"):
        botmix_rows.append(
            {
                "condition": condition_name,
                "bot_mix": str(frame["bot_mix"].iloc[0]),
                "challenge_f1_mean": float(frame["standard_challenge_f1"].mean()),
                "challenge_f1_std": float(frame["standard_challenge_f1"].std(ddof=0)),
                "outcome_accuracy_mean": float(frame["standard_outcome_accuracy"].mean()),
                "outcome_accuracy_std": float(frame["standard_outcome_accuracy"].std(ddof=0)),
                "outcome_score_mean": float(frame["standard_outcome_score"].mean()),
                "outcome_score_std": float(frame["standard_outcome_score"].std(ddof=0)),
            }
        )
    table_botmix = pd.DataFrame(botmix_rows).sort_values("condition")
    table_botmix.to_csv(tables_dir / "table_botmix_comparison.csv", index=False)

    ranking = table_botmix.copy()
    ranking["weighted_score"] = (
        0.6 * ranking["outcome_accuracy_mean"] + 0.4 * ranking["challenge_f1_mean"]
    )
    ranking = ranking.sort_values("weighted_score", ascending=False)
    ranking.to_csv(tables_dir / "table_weighted_ranking.csv", index=False)

    table_belief_ablation = _mode_ablation_table(
        seed_condition,
        left_prefix="standard",
        right_prefix="strict",
        left_label="Standard",
        right_label="Strict no-duplicate-hand",
    )
    table_belief_ablation.to_csv(tables_dir / "table_ablation_belief_mode.csv", index=False)

    table_advisor_ablation = _mode_ablation_table(
        seed_condition,
        left_prefix="standard",
        right_prefix="off",
        left_label="Advisor On",
        right_label="Advisor Off (never challenge)",
    )
    table_advisor_ablation.to_csv(tables_dir / "table_ablation_advisor_on_off.csv", index=False)

    significance = _significance_notes(seed_condition)
    (tables_dir / "significance_notes.json").write_text(json.dumps(significance, indent=2))

    return [
        "table_advisor_main.csv",
        "table_botmix_comparison.csv",
        "table_weighted_ranking.csv",
        "table_ablation_belief_mode.csv",
        "table_ablation_advisor_on_off.csv",
        "seed_condition_metrics.csv",
        "significance_notes.json",
    ]


def _mode_ablation_table(seed_condition, *, left_prefix, right_prefix, left_label, right_label):
    pd, _np = _import_tabular()
    rows = []
    for label, prefix in ((left_label, left_prefix), (right_label, right_prefix)):
        rows.append(
            {
                "mode": label,
                "challenge_f1_mean": float(seed_condition[f"{prefix}_challenge_f1"].mean()),
                "challenge_f1_std": float(seed_condition[f"{prefix}_challenge_f1"].std(ddof=0)),
                "outcome_accuracy_mean": float(seed_condition[f"{prefix}_outcome_accuracy"].mean()),
                "outcome_accuracy_std": float(
                    seed_condition[f"{prefix}_outcome_accuracy"].std(ddof=0)
                ),
                "outcome_score_mean": float(seed_condition[f"{prefix}_outcome_score"].mean()),
                "outcome_score_std": float(seed_condition[f"{prefix}_outcome_score"].std(ddof=0)),
            }
        )

    delta = {
        "mode": f"Delta ({right_label} - {left_label})",
        "challenge_f1_mean": rows[1]["challenge_f1_mean"] - rows[0]["challenge_f1_mean"],
        "challenge_f1_std": 0.0,
        "outcome_accuracy_mean": (
            rows[1]["outcome_accuracy_mean"] - rows[0]["outcome_accuracy_mean"]
        ),
        "outcome_accuracy_std": 0.0,
        "outcome_score_mean": rows[1]["outcome_score_mean"] - rows[0]["outcome_score_mean"],
        "outcome_score_std": 0.0,
    }
    rows.append(delta)
    return pd.DataFrame(rows)


def _significance_notes(seed_condition):
    notes = {}
    for metric in ("challenge_f1", "outcome_accuracy", "outcome_score"):
        strict_delta = (
            seed_condition[f"strict_{metric}"] - seed_condition[f"standard_{metric}"]
        ).tolist()
        off_delta = (
            seed_condition[f"standard_{metric}"] - seed_condition[f"off_{metric}"]
        ).tolist()
        strict_stats = _summary_stats(strict_delta)
        off_stats = _summary_stats(off_delta)
        notes[f"strict_minus_standard_{metric}"] = {
            "mean_delta": strict_stats["mean"],
            "ci95_low": strict_stats["ci95_low"],
            "ci95_high": strict_stats["ci95_high"],
            "likely_nonzero": strict_stats["ci95_low"] > 0.0
            or strict_stats["ci95_high"] < 0.0,
        }
        notes[f"advisor_on_minus_off_{metric}"] = {
            "mean_delta": off_stats["mean"],
            "ci95_low": off_stats["ci95_low"],
            "ci95_high": off_stats["ci95_high"],
            "likely_nonzero": off_stats["ci95_low"] > 0.0
            or off_stats["ci95_high"] < 0.0,
        }
    return notes


def _build_policy_tables(pd, bc_train_metrics, bc_benchmark, dqn_summary, dqn_dir):
    benchmark = dqn_summary.get("benchmark", {})
    dqn_policy = benchmark.get("dqn_policy", {})
    random_policy = benchmark.get("random_policy", {})
    bc_policy = benchmark.get("behavior_cloning", {}) or bc_benchmark.get("behavior_cloning", {})

    policy_rows = [
        {
            "policy": "DQN",
            "avg_reward_per_episode": float(dqn_policy.get("avg_reward_per_episode", 0.0)),
            "avg_reward_per_step": float(dqn_policy.get("avg_reward_per_step", 0.0)),
            "action_match_rate": float(dqn_policy.get("action_match_rate", 0.0)),
        },
        {
            "policy": "Random",
            "avg_reward_per_episode": float(random_policy.get("avg_reward_per_episode", 0.0)),
            "avg_reward_per_step": float(random_policy.get("avg_reward_per_step", 0.0)),
            "action_match_rate": float(random_policy.get("action_match_rate", 0.0)),
        },
        {
            "policy": "Behavior Cloning",
            "avg_reward_per_episode": float(bc_policy.get("avg_reward_per_episode", 0.0)),
            "avg_reward_per_step": float(bc_policy.get("avg_reward_per_step", 0.0)),
            "action_match_rate": float(bc_policy.get("action_match_rate", 0.0)),
        },
    ]

    for key, label in POLICY_LABELS.items():
        payload = benchmark.get(key, {}) or bc_benchmark.get(key, {})
        if not payload:
            continue
        policy_rows.append(
            {
                "policy": label,
                "avg_reward_per_episode": float(payload.get("avg_reward_per_episode", 0.0)),
                "avg_reward_per_step": float(payload.get("avg_reward_per_step", 0.0)),
                "action_match_rate": float(payload.get("action_match_rate", 0.0)),
            }
        )

    policy_table = pd.DataFrame(policy_rows)

    train_metrics = dqn_summary.get("train_metrics", {})
    state_summary_path = Path(dqn_dir) / "train_state_summary.json"
    state_observations = 0
    if state_summary_path.exists():
        state_observations = int(json.loads(state_summary_path.read_text()).get("observations", 0))

    training_table = pd.DataFrame(
        [
            {
                "episodes": int(train_metrics.get("episodes", 0)),
                "avg_episode_reward": float(train_metrics.get("avg_episode_reward", 0.0)),
                "avg_episode_steps": float(train_metrics.get("avg_episode_steps", 0.0)),
                "final_epsilon": float(train_metrics.get("final_epsilon", 0.0)),
                "state_observations": int(state_observations),
                "bc_accuracy": float(bc_train_metrics.get("accuracy", 0.0)),
                "bc_f1_macro": float(bc_train_metrics.get("f1_macro", 0.0)),
            }
        ]
    )
    return policy_table, training_table


def _copy_fallback_figures(target_dir, *, fallback_dir=None):
    fallback = Path(fallback_dir) if fallback_dir else None
    figure_names = [
        "fig_advisor_multiseed_metrics.png",
        "fig_botmix_comparison.png",
        "fig_dqn_reward_curve.png",
        "fig_policy_benchmark.png",
    ]
    if fallback is None or not fallback.exists():
        return []

    copied = []
    for name in figure_names:
        source = fallback / name
        target = Path(target_dir) / name
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(name)
    return copied


def _build_figures(*, tables_dir, figures_dir, dqn_dir, fallback_dir=None):
    pd, _np = _import_tabular()
    try:
        plt = _import_matplotlib()
    except Exception:
        return _copy_fallback_figures(figures_dir, fallback_dir=fallback_dir)
    if plt is None:
        return _copy_fallback_figures(figures_dir, fallback_dir=fallback_dir)

    main = pd.read_csv(tables_dir / "table_advisor_main.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(main["metric"], main["mean"], yerr=main["std"], capsize=4)
    ax.set_title("Advisor Multi-Seed Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_advisor_multiseed_metrics.png", dpi=150)
    plt.close(fig)

    botmix = pd.read_csv(tables_dir / "table_botmix_comparison.csv")
    x = list(range(len(botmix)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - 0.18 for i in x], botmix["challenge_f1_mean"], width=0.36, label="challenge_f1")
    ax.bar(
        [i + 0.18 for i in x],
        botmix["outcome_accuracy_mean"],
        width=0.36,
        label="outcome_accuracy",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(botmix["condition"], rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title("Bot-Mix Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_botmix_comparison.png", dpi=150)
    plt.close(fig)

    history_path = Path(dqn_dir) / "train_episode_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        if not history.empty:
            history["rolling_reward_10"] = history["reward"].rolling(
                window=10,
                min_periods=1,
            ).mean()
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(history["episode"], history["reward"], alpha=0.4, label="episode reward")
            ax.plot(
                history["episode"],
                history["rolling_reward_10"],
                linewidth=2,
                label="rolling mean (10)",
            )
            ax.set_title("DQN Reward Curve")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures_dir / "fig_dqn_reward_curve.png", dpi=150)
            plt.close(fig)

    policy = pd.read_csv(tables_dir / "table_policy_benchmark.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(policy["policy"], policy["avg_reward_per_episode"])
    ax.set_title("Policy Benchmark")
    ax.set_ylabel("Avg Reward / Episode")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_policy_benchmark.png", dpi=150)
    plt.close(fig)

    return [
        "fig_advisor_multiseed_metrics.png",
        "fig_botmix_comparison.png",
        "fig_dqn_reward_curve.png",
        "fig_policy_benchmark.png",
    ]


def _summary_stats(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / n
    std = math.sqrt(max(variance, 0.0))
    stderr = std / math.sqrt(n) if n > 0 else 0.0
    margin = 1.96 * stderr
    return {
        "mean": float(mean),
        "std": float(std),
        "ci95_low": float(mean - margin),
        "ci95_high": float(mean + margin),
    }


def _write_csv(path, *, fieldnames, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _import_tabular():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Final experiments require pandas and numpy. Install them in your environment."
        ) from exc
    return pd, np


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _parse_seeds(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_args():
    parser = argparse.ArgumentParser(description="Run final Coup research experiment matrix")
    parser.add_argument(
        "--out-root",
        default="data/research/final_runs",
        help="Base output directory",
    )
    parser.add_argument(
        "--seeds",
        default="101,202,303,404,505",
        help="Comma-separated seed list",
    )
    parser.add_argument("--games-per-condition", type=int, default=50)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--dqn-episodes", type=int, default=100)
    parser.add_argument("--dqn-seed", type=int, default=42)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip matplotlib figure generation",
    )
    parser.add_argument(
        "--figure-fallback-dir",
        default=None,
        help="Optional directory to copy fallback figure files from",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    manifest = run_final_experiments(
        args.out_root,
        seeds=_parse_seeds(args.seeds),
        games_per_condition=args.games_per_condition,
        players=args.players,
        max_turns=args.max_turns,
        dqn_episodes=args.dqn_episodes,
        dqn_seed=args.dqn_seed,
        skip_figures=args.skip_figures,
        figure_fallback_dir=args.figure_fallback_dir,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
