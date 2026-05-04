import type { CoupEvent } from "../api/client";
import { describeEvent } from "../lib/game";

interface Props {
  events: CoupEvent[];
  currentIndex: number;
  onJump: (index: number) => void;
}

export function EventLog({ events, currentIndex, onJump }: Props) {
  return (
    <div className="panel p-3">
      <div className="chip mb-2">Event log</div>
      <div className="max-h-[340px] overflow-y-auto pr-1">
        {events.map((e, i) => {
          const isCurrent = i === currentIndex - 1;
          const isPast = i < currentIndex;
          return (
            <button
              key={i}
              onClick={() => onJump(i + 1)}
              className={`block w-full text-left px-2 py-1 rounded text-sm transition ${
                isCurrent
                  ? "bg-poker-red/25 border border-poker-red/60"
                  : isPast
                    ? "text-poker-ink hover:bg-white/5"
                    : "text-poker-muted/70 hover:bg-white/5"
              }`}
            >
              <span className="font-mono text-xs text-poker-muted mr-2">
                #{String(i + 1).padStart(2, "0")}
              </span>
              {describeEvent(e)}
            </button>
          );
        })}
      </div>
    </div>
  );
}
