import type { PublicState, Role } from "../api/client";
import { ROLES } from "../lib/game";

interface Props {
  publicState: PublicState;
  players: string[];
  remaining: Record<Role, number>;
}

export function PublicStateTable({ publicState, players, remaining }: Props) {
  return (
    <div className="panel p-3 space-y-3">
      <div>
        <div className="chip mb-2">Public state</div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-poker-muted">
              <th className="text-left py-1 pr-3">Player</th>
              <th className="text-right py-1 px-2">Coins</th>
              <th className="text-right py-1 px-2">Influence</th>
            </tr>
          </thead>
          <tbody>
            {players.map((name) => {
              const p = publicState.players[name];
              return (
                <tr key={name} className="border-t border-poker-line/30">
                  <td className="py-1 pr-3 font-display">{name}</td>
                  <td className="py-1 px-2 text-right font-mono">{p?.coins ?? 0}</td>
                  <td className="py-1 px-2 text-right font-mono">
                    {p?.influence_alive ?? 0}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div>
        <div className="chip mb-2">Revealed dead / remaining copies</div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-poker-muted">
              <th className="text-left py-1 pr-3">Role</th>
              <th className="text-right py-1 px-2">Dead</th>
              <th className="text-right py-1 px-2">Remaining</th>
            </tr>
          </thead>
          <tbody>
            {ROLES.map((r) => (
              <tr key={r} className="border-t border-poker-line/30">
                <td className="py-1 pr-3 font-display">{r}</td>
                <td className="py-1 px-2 text-right font-mono">
                  {publicState.revealed_dead[r] ?? 0}
                </td>
                <td className="py-1 px-2 text-right font-mono">{remaining[r]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
