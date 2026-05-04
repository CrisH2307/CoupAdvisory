from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from coup.sim.metrics import build_trace, calibration_bin_labels, summarize_game
from coup.sim.sim import simulate_games


def main():
    parser = argparse.ArgumentParser(description="Simulate synthetic Coup games")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--out", default="data/sim")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--bot-mix",
        default=None,
        help=(
            "Comma-separated bot styles "
            "(honest,bluffer,aggressive,cautious,rule_based,belief_ev,pressure)"
        ),
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--strict-no-duplicate-hand", action="store_true")
    parser.add_argument(
        "--advisor-policy",
        choices=["advisor", "never_challenge", "always_challenge"],
        default="advisor",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bot_mix = None
    if args.bot_mix:
        bot_mix = [part.strip() for part in args.bot_mix.split(",") if part.strip()]

    results = simulate_games(
        args.games,
        players=args.players,
        seed=args.seed,
        bot_mix=bot_mix,
        max_turns=args.max_turns,
    )

    summary_rows = []
    bin_labels = calibration_bin_labels()
    for index, result in enumerate(results, start=1):
        game_path = out_dir / f"game_{index}.json"
        game_path.write_text(json.dumps(result.events, indent=2))

        metrics = summarize_game(
            events=result.events,
            players=result.players,
            winner=result.winner,
            turns=result.turns,
            strict_no_duplicate_hand=args.strict_no_duplicate_hand,
            advisor_policy=args.advisor_policy,
        )

        if args.trace:
            trace = build_trace(
                result.events,
                result.players,
                strict_no_duplicate_hand=args.strict_no_duplicate_hand,
            )
            (out_dir / f"trace_{index}.json").write_text(json.dumps(trace, indent=2))

        row = {"game": index}
        row.update(asdict(metrics))

        histogram = row.pop("reveal_probability_bins", {})
        for label in bin_labels:
            key = f"reveal_prob_bin_{label.replace('.', '_').replace('-', '_')}"
            row[key] = int(histogram.get(label, 0))

        summary_rows.append(row)

    summary_path = out_dir / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    payload = {
        "out_dir": str(out_dir),
        "games": len(results),
        "summary_csv": str(summary_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
