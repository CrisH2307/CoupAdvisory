import type { PublicState } from "../api/client";

interface Props {
  players: string[];
  publicState: PublicState;
  activePlayer?: string | null;
}

const SEAT_POSITIONS = [
  { top: "78%", left: "50%" },
  { top: "60%", left: "12%" },
  { top: "22%", left: "18%" },
  { top: "8%", left: "50%" },
  { top: "22%", left: "82%" },
  { top: "60%", left: "88%" },
];

export function PokerTable({ players, publicState, activePlayer }: Props) {
  return (
    <div className="relative w-full h-[440px] rounded-2xl overflow-hidden border border-poker-line">
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(130% 90% at 50% 20%, rgba(60,68,106,0.32) 0%, rgba(15,19,40,0.18) 55%, rgba(11,15,31,0.88) 100%)",
        }}
      />
      <div
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border-[10px] border-[#1f2430] shadow-2xl"
        style={{
          width: "min(83%, 760px)",
          height: "min(60%, 300px)",
          borderRadius: "50% / 44%",
          background:
            "radial-gradient(ellipse at center, #0f8553 0%, #0a6f44 60%, #085a37 100%)",
          boxShadow:
            "inset 0 0 28px rgba(0,0,0,0.42), 0 18px 34px rgba(0,0,0,0.45)",
        }}
      >
        <div className="absolute inset-0 flex items-center justify-center font-display text-2xl text-white/20 tracking-widest">
          COUP
        </div>
      </div>

      {players.map((name, idx) => {
        const pos = SEAT_POSITIONS[idx % SEAT_POSITIONS.length];
        const p = publicState.players[name];
        const isActive = activePlayer === name;
        const alive = (p?.influence_alive ?? 0) > 0;
        return (
          <div
            key={name}
            className="absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: pos.left, top: pos.top }}
          >
            <div
              className={`panel-sm px-3 py-2 min-w-[110px] text-center transition-all ${
                isActive ? "ring-2 ring-poker-red scale-105" : ""
              } ${!alive ? "opacity-40" : ""}`}
            >
              <div className="font-display text-sm text-poker-ink">{name}</div>
              <div className="text-xs text-poker-muted mt-1">
                <span className="text-yellow-300">●</span>{" "}
                {p?.coins ?? 0} coins
              </div>
              <div className="mt-1 flex justify-center gap-1">
                {Array.from({ length: 2 }).map((_, i) => (
                  <span
                    key={i}
                    className={`inline-block w-3 h-4 rounded-sm border ${
                      i < (p?.influence_alive ?? 0)
                        ? "bg-poker-red/80 border-red-300"
                        : "bg-slate-700 border-slate-500"
                    }`}
                  />
                ))}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
