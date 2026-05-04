import type { AdvisorRecommendation } from "../api/client";

interface Props {
  recommendation: AdvisorRecommendation | null;
  claimant: string | null;
  role: string | null;
  loading?: boolean;
}

export function AdvisorCard({ recommendation, claimant, role, loading }: Props) {
  const isChallenge = recommendation?.recommendation === "Challenge";
  return (
    <div className="panel p-4">
      <div className="chip mb-2">Advisor desk</div>
      {!claimant && !recommendation && (
        <div className="text-poker-muted text-sm">
          Step forward to a claim event (action or block) to see a recommendation.
        </div>
      )}
      {claimant && (
        <div className="text-sm text-poker-muted mb-2">
          Claim: <span className="font-display text-poker-ink">{claimant}</span> → {role}
        </div>
      )}
      {loading && <div className="text-poker-muted text-sm">computing…</div>}
      {recommendation && (
        <>
          <div
            className={`font-display text-2xl mb-1 ${
              isChallenge ? "text-poker-red" : "text-emerald-300"
            }`}
          >
            {recommendation.recommendation}
          </div>
          <div className="text-xs text-poker-muted mb-2">
            {recommendation.explanation}
          </div>
          <div className="flex gap-4 text-sm">
            <div>
              <div className="text-poker-muted text-xs uppercase tracking-wider">
                p(truth)
              </div>
              <div className="font-mono text-lg">
                {(recommendation.p_truth * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-poker-muted text-xs uppercase tracking-wider">
                threshold
              </div>
              <div className="font-mono text-lg">
                {(recommendation.threshold * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
