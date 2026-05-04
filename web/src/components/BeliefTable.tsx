import type { BeliefState, Role } from "../api/client";
import { ROLES } from "../lib/game";

interface Props {
  belief: BeliefState;
  players: string[];
}

function heatColor(p: number): string {
  const clamped = Math.max(0, Math.min(1, p));
  const hue = 210 - clamped * 210;
  const lightness = 20 + clamped * 35;
  return `hsl(${hue}, 70%, ${lightness}%)`;
}

export function BeliefTable({ belief, players }: Props) {
  return (
    <div className="panel p-3">
      <div className="chip mb-2">Hidden role probabilities</div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-poker-muted">
              <th className="text-left py-1 pr-3">Player</th>
              {ROLES.map((r) => (
                <th key={r} className="text-right py-1 px-2">
                  {r}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {players.map((name) => {
              const row = belief.probabilities[name] ?? {};
              return (
                <tr key={name} className="border-t border-poker-line/30">
                  <td className="py-1 pr-3 font-display">{name}</td>
                  {ROLES.map((r) => {
                    const p = Number(row[r as Role] ?? 0);
                    return (
                      <td
                        key={r}
                        className="py-1 px-2 text-right font-mono"
                        style={{
                          backgroundColor: heatColor(p),
                          color: p > 0.5 ? "#fff" : "#e2ecff",
                        }}
                      >
                        {(p * 100).toFixed(0)}%
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
