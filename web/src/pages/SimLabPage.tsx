import { useEffect, useState } from "react";
import { api, type SimRunResponse } from "../api/client";
import { MetricsPanel } from "../components/MetricsPanel";

export function SimLabPage() {
  const [players, setPlayers] = useState(4);
  const [games, setGames] = useState(20);
  const [seed, setSeed] = useState(0);
  const [botStyles, setBotStyles] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([
    "honest",
    "bluffer",
    "aggressive",
    "cautious",
  ]);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SimRunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listBots()
      .then((r) => setBotStyles(r.styles))
      .catch(() => setBotStyles([]));
  }, []);

  const toggle = (s: string) =>
    setSelected((prev) =>
      prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s],
    );

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const r = await api.runSim({
        players,
        games,
        seed,
        bot_mix: selected.length > 0 ? selected : null,
      });
      setResult(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-4 p-4 max-w-[1500px] mx-auto">
      <div className="panel p-4">
        <div className="font-display text-2xl mb-3">Simulation Lab</div>
        <div className="flex flex-wrap gap-4 items-end">
          <label className="flex flex-col text-sm">
            <span className="text-poker-muted text-xs uppercase tracking-wider">
              Players
            </span>
            <input
              type="number"
              min={2}
              max={6}
              value={players}
              onChange={(e) => setPlayers(Number(e.target.value))}
              className="btn bg-poker-bg2 font-mono w-24"
            />
          </label>
          <label className="flex flex-col text-sm">
            <span className="text-poker-muted text-xs uppercase tracking-wider">
              Games
            </span>
            <input
              type="number"
              min={1}
              max={500}
              value={games}
              onChange={(e) => setGames(Number(e.target.value))}
              className="btn bg-poker-bg2 font-mono w-24"
            />
          </label>
          <label className="flex flex-col text-sm">
            <span className="text-poker-muted text-xs uppercase tracking-wider">
              Seed
            </span>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="btn bg-poker-bg2 font-mono w-24"
            />
          </label>
          <div className="flex flex-col text-sm">
            <span className="text-poker-muted text-xs uppercase tracking-wider mb-1">
              Bot mix
            </span>
            <div className="flex flex-wrap gap-1">
              {botStyles.map((s) => (
                <button
                  key={s}
                  onClick={() => toggle(s)}
                  className={`chip cursor-pointer ${
                    selected.includes(s)
                      ? "!bg-poker-red/30 !border-poker-red"
                      : ""
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
          <button className="btn btn-primary" onClick={run} disabled={running}>
            {running ? "Running…" : "Run batch"}
          </button>
        </div>
        {error && <div className="text-poker-red text-sm mt-2">{error}</div>}
      </div>

      <MetricsPanel />

      {result && (
        <div className="panel p-3">
          <div className="chip mb-2">Batch results ({result.games.length} games)</div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-poker-muted">
                  <th className="text-left py-1 pr-3">#</th>
                  <th className="text-left py-1 px-2">Winner</th>
                  <th className="text-right py-1 px-2">Turns</th>
                  <th className="text-right py-1 px-2">Challenges</th>
                  <th className="text-right py-1 px-2">Advisor F1</th>
                </tr>
              </thead>
              <tbody>
                {result.summary_rows.map((row, i) => (
                  <tr key={i} className="border-t border-poker-line/30">
                    <td className="py-1 pr-3 font-mono">{row.game}</td>
                    <td className="py-1 px-2 font-display">{String(row.winner ?? "—")}</td>
                    <td className="py-1 px-2 text-right font-mono">
                      {String(row.turns ?? "—")}
                    </td>
                    <td className="py-1 px-2 text-right font-mono">
                      {String(row.challenges ?? "—")}
                    </td>
                    <td className="py-1 px-2 text-right font-mono">
                      {typeof row.advisor_challenge_f1 === "number"
                        ? row.advisor_challenge_f1.toFixed(3)
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
