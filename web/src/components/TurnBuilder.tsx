import { useEffect, useMemo, useState } from "react";
import {
  api,
  type AdvisorRecommendation,
  type CoupEvent,
  type Role,
} from "../api/client";
import { ACTION_CLAIM } from "../lib/game";

type Style = "Conservative" | "Balanced" | "Aggressive";

const ACTIONS = [
  "Income",
  "Foreign Aid",
  "Tax",
  "Steal",
  "Assassinate",
  "Coup",
  "Exchange",
] as const;
type Action = (typeof ACTIONS)[number];

const ROLES_ARR: Role[] = ["Duke", "Assassin", "Captain", "Ambassador", "Contessa"];

const BLOCK_ROLES: Record<string, Role[]> = {
  "Foreign Aid": ["Duke"],
  Steal: ["Captain", "Ambassador"],
  Assassination: ["Contessa"],
};

type Step =
  | "pickAction"
  | "challengePrompt"
  | "challengeOutcome"
  | "blockPrompt"
  | "blockChallengePrompt"
  | "blockChallengeOutcome"
  | "review";

interface Draft {
  actor: string;
  action: Action;
  target: string;
  challenger: string | null;
  challengeResult: "win" | "lose" | null;
  challengeReveal: Role | null;
  blocker: string | null;
  blockRole: Role | null;
  blockChallenger: string | null;
  blockChallengeResult: "win" | "lose" | null;
  blockChallengeReveal: Role | null;
}

const emptyDraft = (actor: string, action: Action = "Income"): Draft => ({
  actor,
  action,
  target: "",
  challenger: null,
  challengeResult: null,
  challengeReveal: null,
  blocker: null,
  blockRole: null,
  blockChallenger: null,
  blockChallengeResult: null,
  blockChallengeReveal: null,
});

interface Props {
  players: string[];
  coinsByPlayer: Record<string, number>;
  aliveByPlayer: Record<string, number>;
  sessionId: string | null;
  style: Style;
  selfPlayer: string | null;
  onAppend: (events: CoupEvent[]) => void | Promise<void>;
}

export function TurnBuilder({
  players,
  coinsByPlayer,
  aliveByPlayer,
  sessionId,
  style,
  selfPlayer,
  onAppend,
}: Props) {
  const alivePlayers = useMemo(
    () => players.filter((p) => (aliveByPlayer[p] ?? 0) > 0),
    [players, aliveByPlayer],
  );

  const [draft, setDraft] = useState<Draft>(() =>
    emptyDraft(alivePlayers[0] ?? ""),
  );
  const [step, setStep] = useState<Step>("pickAction");
  const [advice, setAdvice] = useState<AdvisorRecommendation | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!alivePlayers.includes(draft.actor) && alivePlayers.length) {
      setDraft((d) => ({ ...d, actor: alivePlayers[0] }));
    }
  }, [alivePlayers, draft.actor]);

  const coins = coinsByPlayer[draft.actor] ?? 0;
  const forcedCoup = coins >= 10;
  useEffect(() => {
    if (forcedCoup && draft.action !== "Coup") {
      setDraft((d) => ({ ...d, action: "Coup" }));
    }
  }, [forcedCoup, draft.action]);

  const needsTarget =
    draft.action === "Coup" ||
    draft.action === "Steal" ||
    draft.action === "Assassinate";
  const canBeBlocked =
    draft.action === "Foreign Aid" ||
    draft.action === "Steal" ||
    draft.action === "Assassinate";
  const canBeChallenged = ACTION_CLAIM[draft.action] !== undefined;

  const actionValid =
    draft.action === "Coup"
      ? coins >= 7
      : draft.action === "Assassinate"
        ? coins >= 3
        : true;

  useEffect(() => {
    if (!sessionId) {
      setAdvice(null);
      return;
    }
    const role = ACTION_CLAIM[draft.action];
    if (!role) {
      setAdvice(null);
      return;
    }
    let cancelled = false;
    api
      .advise(sessionId, draft.actor, role, draft.action, null, style)
      .then((r) => {
        if (!cancelled) setAdvice(r);
      })
      .catch(() => setAdvice(null));
    return () => {
      cancelled = true;
    };
  }, [sessionId, draft.actor, draft.action, style]);

  const otherAlive = alivePlayers.filter((p) => p !== draft.actor);
  const blockableOthers = alivePlayers.filter(
    (p) => p !== draft.actor && (!needsTarget || p === draft.target || true),
  );

  const resetFlow = () => {
    setDraft((d) => emptyDraft(d.actor, d.action));
    setStep("pickAction");
    setError(null);
  };

  const advanceFromPickAction = () => {
    setError(null);
    if (!draft.actor) return setError("Pick an actor.");
    if (needsTarget && !draft.target) return setError(`Pick a target.`);
    if (!actionValid) return setError(`${draft.actor} needs more coins.`);

    if (draft.action === "Income" || draft.action === "Coup") {
      setStep("review");
      return;
    }
    if (canBeChallenged) {
      setStep("challengePrompt");
      return;
    }
    if (canBeBlocked) {
      setStep("blockPrompt");
      return;
    }
    setStep("review");
  };

  const selectChallenger = (who: string | null) => {
    if (who === null) {
      setDraft((d) => ({
        ...d,
        challenger: null,
        challengeResult: null,
        challengeReveal: null,
      }));
      if (canBeBlocked) setStep("blockPrompt");
      else setStep("review");
      return;
    }
    setDraft((d) => ({ ...d, challenger: who }));
    setStep("challengeOutcome");
  };

  const selectChallengeOutcome = (
    result: "win" | "lose",
    revealed: Role,
  ) => {
    setDraft((d) => ({
      ...d,
      challengeResult: result,
      challengeReveal: revealed,
    }));
    if (result === "win") {
      setStep("review");
      return;
    }
    if (canBeBlocked) setStep("blockPrompt");
    else setStep("review");
  };

  const selectBlock = (blocker: string | null, role: Role | null) => {
    if (!blocker || !role) {
      setDraft((d) => ({
        ...d,
        blocker: null,
        blockRole: null,
        blockChallenger: null,
        blockChallengeResult: null,
        blockChallengeReveal: null,
      }));
      setStep("review");
      return;
    }
    setDraft((d) => ({ ...d, blocker, blockRole: role }));
    setStep("blockChallengePrompt");
  };

  const selectBlockChallenger = (who: string | null) => {
    if (!who) {
      setDraft((d) => ({ ...d, blockChallenger: null }));
      setStep("review");
      return;
    }
    setDraft((d) => ({ ...d, blockChallenger: who }));
    setStep("blockChallengeOutcome");
  };

  const selectBlockChallengeOutcome = (
    result: "win" | "lose",
    revealed: Role,
  ) => {
    setDraft((d) => ({
      ...d,
      blockChallengeResult: result,
      blockChallengeReveal: revealed,
    }));
    setStep("review");
  };

  const buildEvents = (): CoupEvent[] | null => {
    const d = draft;
    const events: CoupEvent[] = [];
    events.push({
      type: "action",
      actor: d.actor,
      action_name: d.action,
      target: needsTarget ? d.target : null,
    });

    if (d.action === "Income") {
      events.push({ type: "coin_change", player: d.actor, delta: 1 });
      return events;
    }

    if (d.action === "Coup") {
      events.push({ type: "coin_change", player: d.actor, delta: -7 });
      if (d.target) {
        events.push({
          type: "influence_lost",
          player: d.target,
          revealed_role: d.challengeReveal ?? "Duke",
        });
      }
      return events;
    }

    if (canBeChallenged && d.challenger && d.challengeResult) {
      const claimed = ACTION_CLAIM[d.action]!;
      events.push({
        type: "challenge",
        challenger: d.challenger,
        challenged: d.actor,
        claimed_role: claimed,
        result: d.challengeResult,
        revealed_role: d.challengeReveal,
      });
      if (d.challengeResult === "win") {
        events.push({
          type: "influence_lost",
          player: d.challenger,
          revealed_role: d.challengeReveal ?? claimed,
        });
      } else {
        events.push({
          type: "influence_lost",
          player: d.actor,
          revealed_role: d.challengeReveal ?? claimed,
        });
        if (d.action === "Assassinate") {
          events.push({ type: "coin_change", player: d.actor, delta: -3 });
        }
        return events;
      }
    }

    if (d.action === "Assassinate") {
      events.push({ type: "coin_change", player: d.actor, delta: -3 });
    }

    if (canBeBlocked && d.blocker && d.blockRole) {
      const blockedAction =
        d.action === "Assassinate" ? "Assassination" : d.action;
      events.push({
        type: "block",
        blocker: d.blocker,
        blocked_action: blockedAction,
        roles_claimed: [d.blockRole],
      });
      if (d.blockChallenger && d.blockChallengeResult) {
        events.push({
          type: "challenge",
          challenger: d.blockChallenger,
          challenged: d.blocker,
          claimed_role: d.blockRole,
          result: d.blockChallengeResult,
          revealed_role: d.blockChallengeReveal,
        });
        if (d.blockChallengeResult === "win") {
          events.push({
            type: "influence_lost",
            player: d.blockChallenger,
            revealed_role: d.blockChallengeReveal ?? d.blockRole,
          });
          return events;
        }
        events.push({
          type: "influence_lost",
          player: d.blocker,
          revealed_role: d.blockChallengeReveal ?? d.blockRole,
        });
      } else {
        return events;
      }
    }

    if (d.action === "Foreign Aid") {
      events.push({ type: "coin_change", player: d.actor, delta: 2 });
    } else if (d.action === "Tax") {
      events.push({ type: "coin_change", player: d.actor, delta: 3 });
    } else if (d.action === "Steal" && d.target) {
      const amount = Math.min(2, coinsByPlayer[d.target] ?? 0);
      if (amount > 0) {
        events.push({ type: "coin_change", player: d.target, delta: -amount });
        events.push({ type: "coin_change", player: d.actor, delta: amount });
      }
    } else if (d.action === "Assassinate" && d.target) {
      events.push({
        type: "influence_lost",
        player: d.target,
        revealed_role: "Duke",
      });
    }
    return events;
  };

  const handleApply = async () => {
    const evs = buildEvents();
    if (!evs) return;
    await onAppend(evs);
    resetFlow();
  };

  const pendingEvents = step === "review" ? buildEvents() ?? [] : [];

  const isYourTurn = selfPlayer && selfPlayer === draft.actor;
  const selfPrefix = isYourTurn ? "(You) " : "";

  const blockRolesForAction =
    BLOCK_ROLES[
      draft.action === "Assassinate" ? "Assassination" : (draft.action as string)
    ] ?? [];

  return (
    <div className="panel p-4 space-y-3">
      <div className="flex items-center flex-wrap gap-2">
        <div className="chip">Turn builder · step: {step}</div>
        {forcedCoup && (
          <div className="chip !border-poker-red !text-poker-red">forced Coup ≥10 coins</div>
        )}
        {advice && step === "pickAction" && (
          <div className="chip !border-emerald-400">
            Advisor: {advice.recommendation} ({(advice.p_truth * 100).toFixed(0)}%)
          </div>
        )}
        {isYourTurn && (
          <div className="chip !border-amber-400">Your turn</div>
        )}
        {step !== "pickAction" && (
          <button className="btn text-xs" onClick={resetFlow}>
            ← Restart turn
          </button>
        )}
      </div>

      {step === "pickAction" && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <label className="flex flex-col text-sm">
            <span className="text-poker-muted text-xs uppercase tracking-wider">Actor</span>
            <select
              value={draft.actor}
              onChange={(e) => setDraft((d) => ({ ...d, actor: e.target.value }))}
              className="btn bg-poker-bg2 cursor-pointer"
            >
              {alivePlayers.map((p) => (
                <option key={p} value={p}>
                  {selfPlayer === p ? "(You) " : ""}
                  {p} ({coinsByPlayer[p] ?? 0}c)
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col text-sm">
            <span className="text-poker-muted text-xs uppercase tracking-wider">Action</span>
            <select
              value={draft.action}
              onChange={(e) =>
                setDraft((d) => ({ ...d, action: e.target.value as Action }))
              }
              className="btn bg-poker-bg2 cursor-pointer"
              disabled={forcedCoup}
            >
              {ACTIONS.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </label>

          {needsTarget && (
            <label className="flex flex-col text-sm">
              <span className="text-poker-muted text-xs uppercase tracking-wider">Target</span>
              <select
                value={draft.target}
                onChange={(e) =>
                  setDraft((d) => ({ ...d, target: e.target.value }))
                }
                className="btn bg-poker-bg2 cursor-pointer"
              >
                <option value="">—</option>
                {otherAlive.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
          )}

          <div className="flex items-end">
            <button className="btn btn-primary w-full" onClick={advanceFromPickAction}>
              Next →
            </button>
          </div>
        </div>
      )}

      {step === "challengePrompt" && (
        <div className="space-y-2">
          <div className="text-sm">
            {selfPrefix}
            {draft.actor} claims <b>{ACTION_CLAIM[draft.action]}</b> with{" "}
            <b>{draft.action}</b>. Does anyone challenge?
          </div>
          <div className="flex flex-wrap gap-2">
            <button className="btn" onClick={() => selectChallenger(null)}>
              No one challenges
            </button>
            {otherAlive.map((p) => (
              <button
                key={p}
                className="btn"
                onClick={() => selectChallenger(p)}
              >
                {selfPlayer === p ? "(You) " : ""}
                {p} challenges
              </button>
            ))}
          </div>
        </div>
      )}

      {step === "challengeOutcome" && (
        <div className="space-y-2">
          <div className="text-sm">
            {draft.challenger} challenges. What does {draft.actor} reveal?
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            {ROLES_ARR.map((r) => {
              const claimed = ACTION_CLAIM[draft.action]!;
              const result: "win" | "lose" = r === claimed ? "lose" : "win";
              return (
                <button
                  key={r}
                  className={`btn ${r === claimed ? "!border-emerald-500" : "!border-poker-red"}`}
                  onClick={() => selectChallengeOutcome(result, r)}
                >
                  {r}
                  <div className="text-[10px] text-poker-muted">
                    {result === "win"
                      ? `bluff → ${draft.actor} loses`
                      : `truth → ${draft.challenger} loses`}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {step === "blockPrompt" && (
        <div className="space-y-2">
          <div className="text-sm">Does anyone block?</div>
          <div className="flex flex-wrap gap-2">
            <button className="btn" onClick={() => selectBlock(null, null)}>
              No block
            </button>
            {blockableOthers.map((p) =>
              blockRolesForAction.map((r) => (
                <button
                  key={`${p}-${r}`}
                  className="btn"
                  onClick={() => selectBlock(p, r)}
                >
                  {selfPlayer === p ? "(You) " : ""}
                  {p} blocks with {r}
                </button>
              )),
            )}
          </div>
        </div>
      )}

      {step === "blockChallengePrompt" && (
        <div className="space-y-2">
          <div className="text-sm">
            {draft.blocker} claims <b>{draft.blockRole}</b> to block. Does {draft.actor} or anyone else challenge?
          </div>
          <div className="flex flex-wrap gap-2">
            <button className="btn" onClick={() => selectBlockChallenger(null)}>
              No challenge — block stands
            </button>
            {alivePlayers
              .filter((p) => p !== draft.blocker)
              .map((p) => (
                <button
                  key={p}
                  className="btn"
                  onClick={() => selectBlockChallenger(p)}
                >
                  {selfPlayer === p ? "(You) " : ""}
                  {p} challenges block
                </button>
              ))}
          </div>
        </div>
      )}

      {step === "blockChallengeOutcome" && draft.blockRole && (
        <div className="space-y-2">
          <div className="text-sm">
            {draft.blockChallenger} challenges the block. What does {draft.blocker} reveal?
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            {ROLES_ARR.map((r) => {
              const claimed = draft.blockRole!;
              const result: "win" | "lose" = r === claimed ? "lose" : "win";
              return (
                <button
                  key={r}
                  className={`btn ${r === claimed ? "!border-emerald-500" : "!border-poker-red"}`}
                  onClick={() => selectBlockChallengeOutcome(result, r)}
                >
                  {r}
                  <div className="text-[10px] text-poker-muted">
                    {result === "win"
                      ? `bluff → block fails`
                      : `truth → challenger loses`}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {step === "review" && (
        <div className="space-y-2">
          <div className="text-sm text-poker-muted">
            {pendingEvents.length} event(s) ready to apply:
          </div>
          <ol className="space-y-1 text-xs font-mono">
            {pendingEvents.map((ev, i) => (
              <li key={i} className="bg-poker-bg2 rounded px-2 py-1">
                {JSON.stringify(ev)}
              </li>
            ))}
          </ol>
          <div className="flex justify-end gap-2">
            <button className="btn" onClick={resetFlow}>
              Discard
            </button>
            <button className="btn btn-primary" onClick={handleApply}>
              Apply turn →
            </button>
          </div>
        </div>
      )}

      {error && <div className="text-poker-red text-sm">{error}</div>}
    </div>
  );
}
