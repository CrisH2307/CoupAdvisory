from __future__ import annotations

from collections import defaultdict

from coup.models import Role


class BluffTracker:
    """
    Tracks per-player role claim history and computes a bluff likelihood
    adjustment factor that can modulate belief updates.

    Adjustment logic:
    - If a player has claimed role R more times than `remaining_copies[R]`,
      at least (claims - remaining) of those were lies. The suspicion score
      rises accordingly.
    - The adjustment_factor returned is in range [0.3, 1.0]:
      1.0 = no suspicion (claims consistent with deck), lower = suspicious.
    """

    def __init__(self):
        # claims[player][role] = number of times player claimed this role
        self.claims: dict[str, dict[Role, int]] = defaultdict(lambda: defaultdict(int))
        # catches[player][role] = times caught bluffing this role
        self.catches: dict[str, dict[Role, int]] = defaultdict(lambda: defaultdict(int))
        # successes[player][role] = times claim was verified true via challenge win
        self.successes: dict[str, dict[Role, int]] = defaultdict(lambda: defaultdict(int))

    def record_claim(self, player: str, role: Role) -> None:
        self.claims[player][role] += 1

    def record_catch(self, player: str, role: Role) -> None:
        """Called when a player is successfully challenged (bluff caught)."""
        self.catches[player][role] += 1

    def record_success(self, player: str, role: Role) -> None:
        """Called when a player wins a challenge (claim proven true)."""
        self.successes[player][role] += 1

    def suspicion_factor(self, player: str, role: Role, remaining_copies: int) -> float:
        """
        Returns a factor in [0.3, 1.0] indicating how suspicious this
        player's claim is based on history.

        1.0 = no extra suspicion
        0.3 = high suspicion (player has been caught bluffing or over-claimed)
        """
        total_claims = self.claims[player].get(role, 0)
        total_catches = self.catches[player].get(role, 0)
        total_successes = self.successes[player].get(role, 0)

        if total_claims == 0:
            return 1.0

        # Hard evidence: caught bluffing this role before
        if total_catches > 0:
            # Each catch reduces trust significantly
            factor = max(0.3, 1.0 - 0.25 * total_catches)
            return float(factor)

        # Soft evidence: claimed more than deck allows
        over_claims = max(0, total_claims - remaining_copies)
        if over_claims > 0:
            factor = max(0.4, 1.0 - 0.15 * over_claims)
            return float(factor)

        # Confirmed truths: slightly boost trust
        if total_successes > 0:
            factor = min(1.0, 1.0 + 0.05 * total_successes)
            return float(factor)

        return 1.0

    def claim_count(self, player: str, role: Role) -> int:
        return self.claims[player].get(role, 0)

    def catch_count(self, player: str, role: Role) -> int:
        return self.catches[player].get(role, 0)

    def to_dict(self) -> dict:
        """Serialize for display/debugging."""
        result = {}
        for player in self.claims:
            result[player] = {
                role.value: {
                    "claims": self.claims[player].get(role, 0),
                    "catches": self.catches[player].get(role, 0),
                    "successes": self.successes[player].get(role, 0),
                }
                for role in Role
                if self.claims[player].get(role, 0) > 0
            }
        return result
