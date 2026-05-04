import { useState } from "react";
import { ManualPlayPage } from "./pages/ManualPlayPage";
import { ReplayPage } from "./pages/ReplayPage";
import { SimLabPage } from "./pages/SimLabPage";

type Tab = "replay" | "manual" | "sim";

export default function App() {
  const [tab, setTab] = useState<Tab>("replay");

  return (
    <div className="min-h-screen">
      <nav className="sticky top-0 z-10 border-b border-poker-line/60 backdrop-blur bg-poker-bg2/70">
        <div className="max-w-[1500px] mx-auto flex items-center gap-4 px-4 py-3">
          <div className="font-display text-xl tracking-widest">COUP</div>
          <div className="flex gap-2">
            <button
              onClick={() => setTab("replay")}
              className={`btn ${tab === "replay" ? "btn-primary" : ""}`}
            >
              Match replay
            </button>
            <button
              onClick={() => setTab("manual")}
              className={`btn ${tab === "manual" ? "btn-primary" : ""}`}
            >
              Manual play
            </button>
            <button
              onClick={() => setTab("sim")}
              className={`btn ${tab === "sim" ? "btn-primary" : ""}`}
            >
              Simulation lab
            </button>
          </div>
          <div className="flex-1" />
          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="text-poker-muted text-xs hover:text-poker-ink"
          >
            API docs ↗
          </a>
        </div>
      </nav>
      {tab === "replay" ? (
        <ReplayPage />
      ) : tab === "manual" ? (
        <ManualPlayPage />
      ) : (
        <SimLabPage />
      )}
    </div>
  );
}
