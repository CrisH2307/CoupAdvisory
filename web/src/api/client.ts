export type Role =
  | "Duke"
  | "Assassin"
  | "Captain"
  | "Ambassador"
  | "Contessa";

export type CoupEvent =
  | { type: "action"; actor: string; action_name: string; target?: string | null }
  | { type: "block"; blocker: string; blocked_action: string; roles_claimed: Role[] }
  | {
      type: "challenge";
      challenger: string;
      challenged: string;
      claimed_role: Role;
      result: "win" | "lose";
      revealed_role?: Role | null;
    }
  | { type: "reveal"; player: string; revealed_role: Role }
  | { type: "coin_change"; player: string; delta: number }
  | { type: "influence_lost"; player: string; revealed_role: Role };

export interface PlayerPublicState {
  coins: number;
  influence_alive: number;
}

export interface PublicState {
  players: Record<string, PlayerPublicState>;
  revealed_dead: Record<Role, number>;
  history: CoupEvent[];
}

export interface BeliefState {
  scores: Record<string, Record<Role, number>>;
  probabilities: Record<string, Record<Role, number>>;
}

export interface GameStateResponse {
  session_id: string;
  player_names: string[];
  public_state: PublicState;
  belief_state: BeliefState;
  remaining_copies: Record<Role, number>;
  strict_no_duplicate_hand: boolean;
  self_seat?: number | null;
  self_hand?: Role[];
}

export interface AdvisorRecommendation {
  recommendation: "Challenge" | "Do not challenge";
  explanation: string;
  p_truth: number;
  threshold: number;
  engine: string;
}

export interface MetricsSnapshot {
  policy: string;
  games: number;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  accuracy?: number | null;
  outcome_accuracy?: number | null;
  extra?: Record<string, unknown>;
}

export interface SimGameResult {
  game_index: number;
  winner: string | null;
  turns: number;
  events: CoupEvent[];
}

export interface SimRunResponse {
  run_id: string;
  games: SimGameResult[];
  summary_rows: Array<Record<string, number | string | null>>;
}

const API_BASE = "";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, `${res.status} ${res.statusText}: ${body}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => fetch(`${API_BASE}/api/health`).then((r) => j<{ ok: boolean }>(r)),

  sampleGame: () =>
    fetch(`${API_BASE}/api/sample-game`).then((r) => j<CoupEvent[]>(r)),

  createGame: (player_names: string[], strict_no_duplicate_hand = false) =>
    fetch(`${API_BASE}/api/games`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ player_names, strict_no_duplicate_hand }),
    }).then((r) => j<GameStateResponse>(r)),

  getGame: (sid: string) =>
    fetch(`${API_BASE}/api/games/${sid}`).then((r) => j<GameStateResponse>(r)),

  applyEvent: (sid: string, event: CoupEvent) =>
    fetch(`${API_BASE}/api/games/${sid}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(event),
    }).then((r) => j<GameStateResponse>(r)),

  replay: (sid: string, events: CoupEvent[]) =>
    fetch(`${API_BASE}/api/games/${sid}/replay`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ events }),
    }).then((r) => j<GameStateResponse>(r)),

  setSelf: (sid: string, seat: number | null, hand: Role[]) =>
    fetch(`${API_BASE}/api/games/${sid}/self`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seat, hand }),
    }).then((r) => j<GameStateResponse>(r)),

  updateSelfHand: (sid: string, hand: Role[]) =>
    fetch(`${API_BASE}/api/games/${sid}/self/hand`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hand }),
    }).then((r) => j<GameStateResponse>(r)),

  resetGame: (sid: string) =>
    fetch(`${API_BASE}/api/games/${sid}/reset`, { method: "POST" }).then((r) =>
      j<GameStateResponse>(r),
    ),

  advise: (
    session_id: string,
    claimant: string,
    claimed_role: Role,
    action_name: string | null,
    block_type: string | null,
    style: "Conservative" | "Balanced" | "Aggressive" = "Balanced",
  ) =>
    fetch(`${API_BASE}/api/advisor/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id,
        claimant,
        claimed_role,
        action_name,
        block_type,
        style,
        engine: "heuristic",
      }),
    }).then((r) => j<AdvisorRecommendation>(r)),

  runSim: (body: {
    players: number;
    games: number;
    seed: number;
    bot_mix?: string[] | null;
    max_turns?: number;
    advisor_policy?: "advisor" | "never_challenge" | "always_challenge";
  }) =>
    fetch(`${API_BASE}/api/sim/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => j<SimRunResponse>(r)),

  listBots: () =>
    fetch(`${API_BASE}/api/sim/bots`).then((r) => j<{ styles: string[] }>(r)),

  metrics: () =>
    fetch(`${API_BASE}/api/research/metrics`).then((r) =>
      j<{ policies: MetricsSnapshot[] }>(r),
    ),
};
