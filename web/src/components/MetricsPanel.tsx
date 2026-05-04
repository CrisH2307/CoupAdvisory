import { useEffect, useState } from "react";
import { api, type MetricsSnapshot } from "../api/client";

function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(3);
}

export function MetricsPanel() {
  const [rows, setRows] = useState<MetricsSnapshot[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .metrics()
      .then((res) => setRows(res.policies))
      .catch((e) => setError(String(e)));
  }, []);

  if (error)
    return (
      <div className="panel p-3 text-poker-muted text-sm">
        Could not load metrics: {error}
      </div>
    );

  return (
    <div className="panel p-3">
      <div className="chip mb-2">Advisor benchmark (heuristic vs ML)</div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-poker-muted">
              <th className="text-left py-1 pr-3">Policy</th>
              <th className="text-right py-1 px-2">Games</th>
              <th className="text-right py-1 px-2">Precision</th>
              <th className="text-right py-1 px-2">Recall</th>
              <th className="text-right py-1 px-2">F1</th>
              <th className="text-right py-1 px-2">Accuracy</th>
              <th className="text-right py-1 px-2">Outcome</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.policy} className="border-t border-poker-line/30">
                <td className="py-1 pr-3 font-display">{r.policy}</td>
                <td className="py-1 px-2 text-right font-mono">{r.games || "—"}</td>
                <td className="py-1 px-2 text-right font-mono">{fmt(r.precision)}</td>
                <td className="py-1 px-2 text-right font-mono">{fmt(r.recall)}</td>
                <td className="py-1 px-2 text-right font-mono">{fmt(r.f1)}</td>
                <td className="py-1 px-2 text-right font-mono">{fmt(r.accuracy)}</td>
                <td className="py-1 px-2 text-right font-mono">
                  {fmt(r.outcome_accuracy)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
