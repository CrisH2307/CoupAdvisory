import { useEffect } from "react";

interface Props {
  index: number;
  total: number;
  playing: boolean;
  speed: number;
  onPlay: () => void;
  onPause: () => void;
  onStep: (delta: number) => void;
  onSeek: (index: number) => void;
  onSpeed: (speed: number) => void;
}

export function PlaybackControls({
  index,
  total,
  playing,
  speed,
  onPlay,
  onPause,
  onStep,
  onSeek,
  onSpeed,
}: Props) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.code === "Space") {
        e.preventDefault();
        if (playing) onPause();
        else onPlay();
      }
      if (e.code === "ArrowRight") onStep(1);
      if (e.code === "ArrowLeft") onStep(-1);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [playing, onPlay, onPause, onStep]);

  return (
    <div className="panel p-3 flex items-center gap-3">
      <button className="btn" onClick={() => onStep(-1)} disabled={index <= 0}>
        ◀
      </button>
      {playing ? (
        <button className="btn btn-primary" onClick={onPause}>
          ⏸ Pause
        </button>
      ) : (
        <button className="btn btn-primary" onClick={onPlay}>
          ▶ Play
        </button>
      )}
      <button
        className="btn"
        onClick={() => onStep(1)}
        disabled={index >= total}
      >
        ▶
      </button>
      <input
        type="range"
        min={0}
        max={total}
        value={index}
        onChange={(e) => onSeek(Number(e.target.value))}
        className="flex-1 accent-poker-red"
      />
      <div className="font-mono text-sm w-16 text-right text-poker-muted">
        {index}/{total}
      </div>
      <select
        value={speed}
        onChange={(e) => onSpeed(Number(e.target.value))}
        className="btn bg-poker-bg2 cursor-pointer"
      >
        <option value={0.5}>0.5×</option>
        <option value={1}>1×</option>
        <option value={2}>2×</option>
        <option value={4}>4×</option>
      </select>
    </div>
  );
}
