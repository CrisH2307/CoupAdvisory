import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  api,
  type AdvisorRecommendation,
  type CoupEvent,
  type GameStateResponse,
} from "../api/client";
import { AdvisorCard } from "../components/AdvisorCard";
import { BeliefTable } from "../components/BeliefTable";
import { EventLog } from "../components/EventLog";
import { PlaybackControls } from "../components/PlaybackControls";
import { PokerTable } from "../components/PokerTable";
import { PublicStateTable } from "../components/PublicStateTable";
import { describeEvent, eventClaim, inferPlayers } from "../lib/game";

type Style = "Conservative" | "Balanced" | "Aggressive";

export function ReplayPage() {
  const [events, setEvents] = useState<CoupEvent[]>([]);
  const [players, setPlayers] = useState<string[]>([]);
  const [session, setSession] = useState<GameStateResponse | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [style, setStyle] = useState<Style>("Balanced");
  const [strict, setStrict] = useState(false);
  const [advice, setAdvice] = useState<AdvisorRecommendation | null>(null);
  const [adviceLoading, setAdviceLoading] = useState(false);
  const [status, setStatus] = useState<string>("");

  const sessionId = session?.session_id ?? null;
  const playingRef = useRef(playing);
  playingRef.current = playing;
  const rebuildAttemptsRef = useRef(0);
  const lastRebuildAtRef = useRef(0);
  const prevSessionIdRef = useRef<string | null>(null);

  const loadEvents = useCallback(
    async (rawEvents: CoupEvent[]) => {
      setStatus("Loading…");
      const ps = inferPlayers(rawEvents);
      if (ps.length === 0) {
        setStatus("No players inferred — empty event log.");
        return;
      }
      const game = await api.createGame(ps, strict);
      rebuildAttemptsRef.current = 0;
      lastRebuildAtRef.current = 0;
      setEvents(rawEvents);
      setPlayers(ps);
      setSession(game);
      setIndex(0);
      setStatus(`Loaded ${rawEvents.length} events, ${ps.length} players.`);
    },
    [strict],
  );

  const loadSample = useCallback(async () => {
    try {
      const raw = await api.sampleGame();
      await loadEvents(raw);
    } catch (e) {
      setStatus(`Failed to load sample: ${e}`);
    }
  }, [loadEvents]);

  const handleUpload = useCallback(
    async (file: File) => {
      const text = await file.text();
      try {
        const parsed = JSON.parse(text) as CoupEvent[];
        await loadEvents(parsed);
      } catch (e) {
        setStatus(`Bad JSON: ${e}`);
      }
    },
    [loadEvents],
  );

  useEffect(() => {
    if (!sessionId || players.length === 0) return;
    const sessionChanged = prevSessionIdRef.current !== sessionId;
    prevSessionIdRef.current = sessionId;
    if (sessionChanged && index === 0) return;
    let cancelled = false;

    const rebuildSession = async (): Promise<GameStateResponse | null> => {
      try {
        const out = await api.resetGame(sessionId);
        rebuildAttemptsRef.current = 0;
        return out;
      } catch (err) {
        if (!(err instanceof ApiError) || err.status !== 404) throw err;
        const now = Date.now();
        if (now - lastRebuildAtRef.current < 4000) {
          rebuildAttemptsRef.current += 1;
        } else {
          rebuildAttemptsRef.current = 1;
        }
        lastRebuildAtRef.current = now;
        if (rebuildAttemptsRef.current > 3) {
          setStatus(
            "Backend session was lost repeatedly — click Load sample / Upload JSON to restart.",
          );
          return null;
        }
        return await api.createGame(players, strict);
      }
    };

    (async () => {
      try {
        const base = await rebuildSession();
        if (cancelled || base === null) return;
        if (index === 0) {
          setSession(base);
          return;
        }
        const slice = events.slice(0, index);
        const replayed = await api.replay(base.session_id, slice);
        if (!cancelled) setSession(replayed);
      } catch (err) {
        if (!cancelled) setStatus(`Session sync failed: ${err}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [index, sessionId, events, players, strict]);

  useEffect(() => {
    if (!playing) return;
    if (index >= events.length) {
      setPlaying(false);
      return;
    }
    const delayMs = 900 / speed;
    const t = setTimeout(() => {
      if (playingRef.current) setIndex((i) => Math.min(events.length, i + 1));
    }, delayMs);
    return () => clearTimeout(t);
  }, [playing, index, events.length, speed]);

  const currentEvent = index > 0 ? events[index - 1] : null;
  const claim = currentEvent ? eventClaim(currentEvent) : null;

  useEffect(() => {
    if (!sessionId || !claim) {
      setAdvice(null);
      return;
    }
    let cancelled = false;
    setAdviceLoading(true);
    api
      .advise(
        sessionId,
        claim.claimant,
        claim.role,
        claim.action ?? null,
        claim.block ?? null,
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
  }, [sessionId, index, claim?.claimant, claim?.role, style]);

  const activePlayer = useMemo(() => {
    if (!currentEvent) return null;
    if (currentEvent.type === "action") return currentEvent.actor;
    if (currentEvent.type === "block") return currentEvent.blocker;
    if (currentEvent.type === "challenge") return currentEvent.challenger;
    return null;
  }, [currentEvent]);

  return (
    <div className="space-y-4 p-4 max-w-[1500px] mx-auto">
      <div className="panel p-4 flex flex-wrap items-center gap-3">
        <div className="font-display text-2xl">COUP Advisor</div>
        <div className="flex-1" />
        <button className="btn" onClick={loadSample}>
          Load sample game
        </button>
        <label className="btn cursor-pointer">
          Upload JSON
          <input
            type="file"
            accept="application/json"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleUpload(f);
            }}
          />
        </label>
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
      </div>

      {status && (
        <div className="text-sm text-poker-muted px-2">{status}</div>
      )}

      {session && (
        <>
          <PokerTable
            players={players}
            publicState={session.public_state}
            activePlayer={activePlayer}
          />

          {currentEvent && (
            <div className="panel p-3 font-display text-lg">
              <span className="chip mr-2">now</span>
              {describeEvent(currentEvent)}
            </div>
          )}

          <PlaybackControls
            index={index}
            total={events.length}
            playing={playing}
            speed={speed}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onStep={(d) =>
              setIndex((i) => Math.max(0, Math.min(events.length, i + d)))
            }
            onSeek={(i) => setIndex(i)}
            onSpeed={setSpeed}
          />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <PublicStateTable
              publicState={session.public_state}
              players={players}
              remaining={session.remaining_copies}
            />
            <BeliefTable belief={session.belief_state} players={players} />
            <AdvisorCard
              recommendation={advice}
              claimant={claim?.claimant ?? null}
              role={claim?.role ?? null}
              loading={adviceLoading}
            />
          </div>

          <EventLog
            events={events}
            currentIndex={index}
            onJump={(i) => setIndex(i)}
          />
        </>
      )}
    </div>
  );
}
