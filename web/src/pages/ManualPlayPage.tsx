import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  type AdvisorRecommendation,
  type CoupEvent,
  type GameStateResponse,
  type Role,
} from "../api/client";
import { AdvisorCard } from "../components/AdvisorCard";
import { BeliefTable } from "../components/BeliefTable";
import { EventLog } from "../components/EventLog";
import { PokerTable } from "../components/PokerTable";
import { PublicStateTable } from "../components/PublicStateTable";
import { SelfSeatCard } from "../components/SelfSeatCard";
import { TurnBuilder } from "../components/TurnBuilder";
import { describeEvent, eventClaim } from "../lib/game";

type Style = "Conservative" | "Balanced" | "Aggressive";

export function ManualPlayPage() {
  const [playerCount, setPlayerCount] = useState(4);
  const [names, setNames] = useState<string[]>(["P1", "P2", "P3", "P4"]);
  const [strict, setStrict] = useState(false);
  const [style, setStyle] = useState<Style>("Balanced");
  const [session, setSession] = useState<GameStateResponse | null>(null);
  const [events, setEvents] = useState<CoupEvent[]>([]);
  const [status, setStatus] = useState<string>("");
  const [advice, setAdvice] = useState<AdvisorRecommendation | null>(null);
  const [adviceLoading, setAdviceLoading] = useState(false);

  const lastEvent = events.length > 0 ? events[events.length - 1] : null;
  const lastClaim = lastEvent ? eventClaim(lastEvent) : null;
  const sessionId = session?.session_id ?? null;

  useEffect(() => {
    if (!sessionId || !lastClaim) {
      setAdvice(null);
      return;
    }
    let cancelled = false;
    setAdviceLoading(true);
    api
      .advise(
        sessionId,
        lastClaim.claimant,
        lastClaim.role,
        lastClaim.action ?? null,
        lastClaim.block ?? null,
        style,
      )
      .then((r) => {
        if (!cancelled) setAdvice(r);
      })
      .catch(() => setAdvice(null))
      .finally(() => {
        if (!cancelled) setAdviceLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, events.length, lastClaim?.claimant, lastClaim?.role, style]);

  const resizeNames = (n: number) => {
    setPlayerCount(n);
    setNames((prev) => {
      const out = [...prev];
      while (out.length < n) out.push(`P${out.length + 1}`);
      out.length = n;
      return out;
    });
  };

  const startGame = useCallback(async () => {
    try {
      const uniq = Array.from(new Set(names.map((n) => n.trim()).filter(Boolean)));
      if (uniq.length !== names.length) {
        setStatus("Player names must be unique and non-empty.");
        return;
      }
      const game = await api.createGame(names, strict);
      setSession(game);
      setEvents([]);
      setStatus(`Started manual game with ${names.length} players.`);
    } catch (e) {
      setStatus(`Failed to start: ${e}`);
    }
  }, [names, strict]);

  const resetGame = useCallback(async () => {
    if (!session) return;
    try {
      const out = await api.resetGame(session.session_id);
      setSession(out);
      setEvents([]);
      setStatus("Game reset.");
    } catch (e) {
      setStatus(`Reset failed: ${e}`);
    }
  }, [session]);

  const appendEvents = useCallback(
    async (evs: CoupEvent[]) => {
      if (!session) return;
      let current: GameStateResponse = session;
      const applied: CoupEvent[] = [];
      try {
        for (const ev of evs) {
          current = await api.applyEvent(session.session_id, ev);
          applied.push(ev);
        }
        setSession(current);
        setEvents((prev) => [...prev, ...applied]);
      } catch (e) {
        setStatus(`Event rejected: ${e}`);
        if (applied.length) {
          setEvents((prev) => [...prev, ...applied]);
          setSession(current);
        }
      }
    },
    [session],
  );

  const undoLast = useCallback(async () => {
    if (!session || events.length === 0) return;
    const sliced = events.slice(0, -1);
    try {
      const base = await api.resetGame(session.session_id);
      const replayed = sliced.length
        ? await api.replay(base.session_id, sliced)
        : base;
      setSession(replayed);
      setEvents(sliced);
    } catch (e) {
      setStatus(`Undo failed: ${e}`);
    }
  }, [session, events]);

  const coinsByPlayer = useMemo(() => {
    const out: Record<string, number> = {};
    if (!session) return out;
    for (const [p, s] of Object.entries(session.public_state.players)) {
      out[p] = s.coins;
    }
    return out;
  }, [session]);

  const aliveByPlayer = useMemo(() => {
    const out: Record<string, number> = {};
    if (!session) return out;
    for (const [p, s] of Object.entries(session.public_state.players)) {
      out[p] = s.influence_alive;
    }
    return out;
  }, [session]);

  return (
    <div className="space-y-4 p-4 max-w-[1500px] mx-auto">
      <div className="panel p-4 space-y-3">
        <div className="font-display text-2xl">Manual Play</div>
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col text-sm">
            <span className="text-poker-muted text-xs uppercase tracking-wider">
              Players
            </span>
            <input
              type="number"
              min={2}
              max={6}
              value={playerCount}
              onChange={(e) => resizeNames(Math.max(2, Math.min(6, Number(e.target.value))))}
              className="btn bg-poker-bg2 font-mono w-24"
            />
          </label>
          {names.map((n, i) => (
            <label key={i} className="flex flex-col text-sm">
              <span className="text-poker-muted text-xs uppercase tracking-wider">
                P{i + 1}
              </span>
              <input
                value={n}
                onChange={(e) =>
                  setNames((prev) => prev.map((x, j) => (j === i ? e.target.value : x)))
                }
                className="btn bg-poker-bg2 font-mono w-28"
              />
            </label>
          ))}
          <label className="flex items-center gap-1 text-sm text-poker-muted">
            <input
              type="checkbox"
              checked={strict}
              onChange={(e) => setStrict(e.target.checked)}
            />
            strict no-dup hand
          </label>
          <select
            value={style}
            onChange={(e) => setStyle(e.target.value as Style)}
            className="btn bg-poker-bg2 cursor-pointer"
          >
            <option value="Conservative">Conservative</option>
            <option value="Balanced">Balanced</option>
            <option value="Aggressive">Aggressive</option>
          </select>
          <button className="btn btn-primary" onClick={startGame}>
            {session ? "Restart with these players" : "Start game"}
          </button>
          {session && (
            <>
              <button className="btn" onClick={resetGame}>
                Reset
              </button>
              <button
                className="btn"
                onClick={undoLast}
                disabled={events.length === 0}
              >
                Undo last
              </button>
            </>
          )}
        </div>
        {status && <div className="text-sm text-poker-muted">{status}</div>}
      </div>

      {session && (
        <>
          <SelfSeatCard
            playerNames={session.player_names}
            selfSeat={session.self_seat ?? null}
            selfHand={session.self_hand}
            locked={events.length > 0}
            onSave={async (seat, hand) => {
              try {
                const out = await api.setSelf(session.session_id, seat, hand);
                setSession(out);
                setStatus(
                  seat === null
                    ? "Cleared self seat."
                    : `You are seated as ${out.player_names[seat]}.`,
                );
              } catch (e) {
                setStatus(`Seat failed: ${e}`);
              }
            }}
            onUpdateHand={async (hand: Role[]) => {
              try {
                const out = await api.updateSelfHand(session.session_id, hand);
                setSession(out);
                setStatus("Your hand updated.");
              } catch (e) {
                setStatus(`Hand update failed: ${e}`);
              }
            }}
          />

          <PokerTable
            players={session.player_names}
            publicState={session.public_state}
            activePlayer={null}
          />

          {events.length > 0 && (
            <div className="panel p-3 font-display text-lg">
              <span className="chip mr-2">last</span>
              {describeEvent(events[events.length - 1])}
            </div>
          )}

          <TurnBuilder
            players={session.player_names}
            coinsByPlayer={coinsByPlayer}
            aliveByPlayer={aliveByPlayer}
            sessionId={session.session_id}
            style={style}
            selfPlayer={
              session.self_seat != null
                ? session.player_names[session.self_seat] ?? null
                : null
            }
            onAppend={appendEvents}
          />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <PublicStateTable
              publicState={session.public_state}
              players={session.player_names}
              remaining={session.remaining_copies}
            />
            <BeliefTable belief={session.belief_state} players={session.player_names} />
            <AdvisorCard
              recommendation={advice}
              claimant={lastClaim?.claimant ?? null}
              role={lastClaim?.role ?? null}
              loading={adviceLoading}
            />
          </div>

          <EventLog events={events} currentIndex={events.length} onJump={() => {}} />
        </>
      )}
    </div>
  );
}
