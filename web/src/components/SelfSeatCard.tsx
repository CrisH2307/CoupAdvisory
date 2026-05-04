import { useState } from "react";
import type { Role } from "../api/client";

const ROLES: Role[] = ["Duke", "Assassin", "Captain", "Ambassador", "Contessa"];

interface Props {
  playerNames: string[];
  selfSeat: number | null | undefined;
  selfHand: Role[] | undefined;
  locked: boolean;
  onSave: (seat: number | null, hand: Role[]) => Promise<void> | void;
  onUpdateHand: (hand: Role[]) => Promise<void> | void;
}

export function SelfSeatCard({
  playerNames,
  selfSeat,
  selfHand,
  locked,
  onSave,
  onUpdateHand,
}: Props) {
  const [seat, setSeat] = useState<number | "">(selfSeat ?? "");
  const [hand, setHand] = useState<[Role | "", Role | ""]>(() => [
    (selfHand?.[0] as Role) ?? "",
    (selfHand?.[1] as Role) ?? "",
  ]);
  const [error, setError] = useState<string | null>(null);

  const pinned = selfSeat !== null && selfSeat !== undefined;
  const pinnedName = pinned ? playerNames[selfSeat!] : null;

  const handleSave = async () => {
    setError(null);
    if (seat === "") {
      await onSave(null, []);
      return;
    }
    const cards = [hand[0], hand[1]].filter((r) => r) as Role[];
    if (cards.length !== 2) {
      setError("Pick both of your starting cards.");
      return;
    }
    await onSave(Number(seat), cards);
  };

  const handleCorrectHand = async () => {
    const cards = [hand[0], hand[1]].filter((r) => r) as Role[];
    if (cards.length === 0) {
      setError("Enter at least one card.");
      return;
    }
    await onUpdateHand(cards);
  };

  return (
    <div className="panel p-3 space-y-2">
      <div className="flex items-center gap-2">
        <div className="chip">Your seat</div>
        {pinned && (
          <div className="chip !border-emerald-400">
            You are {pinnedName} — hand: {(selfHand ?? []).join(" + ") || "?"}
          </div>
        )}
      </div>

      {!locked ? (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-2 items-end">
          <label className="flex flex-col text-sm">
            <span className="text-poker-muted text-xs uppercase tracking-wider">
              Which seat are you?
            </span>
            <select
              value={seat === "" ? "" : String(seat)}
              onChange={(e) =>
                setSeat(e.target.value === "" ? "" : Number(e.target.value))
              }
              className="btn bg-poker-bg2 cursor-pointer"
            >
              <option value="">Neutral observer</option>
              {playerNames.map((name, idx) => (
                <option key={idx} value={idx}>
                  {name} (seat {idx + 1})
                </option>
              ))}
            </select>
          </label>

          {seat !== "" && (
            <>
              <label className="flex flex-col text-sm">
                <span className="text-poker-muted text-xs uppercase tracking-wider">
                  Your card 1
                </span>
                <select
                  value={hand[0]}
                  onChange={(e) =>
                    setHand(([, b]) => [e.target.value as Role, b])
                  }
                  className="btn bg-poker-bg2 cursor-pointer"
                >
                  <option value="">—</option>
                  {ROLES.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col text-sm">
                <span className="text-poker-muted text-xs uppercase tracking-wider">
                  Your card 2
                </span>
                <select
                  value={hand[1]}
                  onChange={(e) =>
                    setHand(([a]) => [a, e.target.value as Role])
                  }
                  className="btn bg-poker-bg2 cursor-pointer"
                >
                  <option value="">—</option>
                  {ROLES.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </label>
            </>
          )}

          <button className="btn btn-primary" onClick={handleSave}>
            Save seat
          </button>
        </div>
      ) : (
        <div className="text-sm text-poker-muted space-y-2">
          <div>
            Seat locked once the game starts. Use the correction row below if you
            lose an influence or exchange cards.
          </div>
          {pinned && (
            <div className="flex flex-wrap items-end gap-2">
              <label className="flex flex-col text-sm">
                <span className="text-poker-muted text-xs uppercase tracking-wider">
                  Card 1
                </span>
                <select
                  value={hand[0]}
                  onChange={(e) =>
                    setHand(([, b]) => [e.target.value as Role, b])
                  }
                  className="btn bg-poker-bg2 cursor-pointer"
                >
                  <option value="">—</option>
                  {ROLES.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col text-sm">
                <span className="text-poker-muted text-xs uppercase tracking-wider">
                  Card 2 (optional)
                </span>
                <select
                  value={hand[1]}
                  onChange={(e) =>
                    setHand(([a]) => [a, e.target.value as Role])
                  }
                  className="btn bg-poker-bg2 cursor-pointer"
                >
                  <option value="">—</option>
                  {ROLES.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </label>
              <button className="btn" onClick={handleCorrectHand}>
                Update my hand
              </button>
            </div>
          )}
        </div>
      )}

      {error && <div className="text-poker-red text-sm">{error}</div>}
    </div>
  );
}
