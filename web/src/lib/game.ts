import type { CoupEvent, Role } from "../api/client";

export const ROLES: Role[] = [
  "Duke",
  "Assassin",
  "Captain",
  "Ambassador",
  "Contessa",
];

export const ACTION_CLAIM: Record<string, Role> = {
  Tax: "Duke",
  Steal: "Captain",
  Assassinate: "Assassin",
  Exchange: "Ambassador",
};

export function inferPlayers(events: CoupEvent[]): string[] {
  const set = new Set<string>();
  for (const e of events) {
    if (e.type === "action") {
      set.add(e.actor);
      if (e.target) set.add(e.target);
    } else if (e.type === "block") {
      set.add(e.blocker);
    } else if (e.type === "challenge") {
      set.add(e.challenger);
      set.add(e.challenged);
    } else if (
      e.type === "reveal" ||
      e.type === "coin_change" ||
      e.type === "influence_lost"
    ) {
      set.add(e.player);
    }
  }
  return Array.from(set).sort();
}

export function describeEvent(e: CoupEvent): string {
  switch (e.type) {
    case "action":
      return e.target
        ? `${e.actor} uses ${e.action_name} on ${e.target}`
        : `${e.actor} uses ${e.action_name}`;
    case "block":
      return `${e.blocker} blocks ${e.blocked_action} claiming ${e.roles_claimed.join("/")}`;
    case "challenge":
      return `${e.challenger} challenges ${e.challenged}'s ${e.claimed_role} → ${e.result}`;
    case "reveal":
      return `${e.player} reveals ${e.revealed_role}`;
    case "influence_lost":
      return `${e.player} loses influence (${e.revealed_role})`;
    case "coin_change":
      return `${e.player} ${e.delta >= 0 ? "+" : ""}${e.delta} coins`;
  }
}

export function eventClaim(
  e: CoupEvent,
): { claimant: string; role: Role; action?: string; block?: string } | null {
  if (e.type === "action") {
    const role = ACTION_CLAIM[e.action_name];
    if (!role) return null;
    return { claimant: e.actor, role, action: e.action_name };
  }
  if (e.type === "block") {
    if (e.roles_claimed.length === 0) return null;
    return {
      claimant: e.blocker,
      role: e.roles_claimed[0],
      block: e.blocked_action,
    };
  }
  return null;
}
