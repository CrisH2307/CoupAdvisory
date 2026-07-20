from __future__ import annotations

import html
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from coup.advisor import ClaimContext, recommend_challenge
from coup.engine import GameEngine, infer_players
from coup.models import ActionEvent, BlockEvent, Role, parse_events
from coup.rules import action_claim_role, block_claim_roles
from coup.sim.sim import simulate_games
from coup.validate import validate_event

SAMPLE_EVENTS_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_game.json"
ACTION_NAMES = ["Income", "Foreign Aid", "Tax", "Steal", "Assassinate", "Coup", "Exchange"]
BLOCKED_ACTIONS = ["Foreign Aid", "Steal", "Assassination"]

st.set_page_config(page_title="Coup Advisor", layout="wide")


def _default_player_names(count):
    size = max(2, min(6, int(count)))
    return [f"P{index + 1}" for index in range(size)]


def _configured_players():
    players = list(st.session_state.get("table_players", []))[:6]
    if players:
        return players
    count = int(st.session_state.get("table_player_count", 4))
    return _default_player_names(count)


def _inject_theme():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {
  --bg-1: #17192a;
  --bg-2: #101427;
  --bg-3: #0b0f1f;
  --ink: #edf3ff;
  --muted: #9eaed1;
  --panel: #131a31;
  --panel-2: #18213b;
  --line: #344974;
  --green-1: #0f8553;
  --green-2: #0a6f44;
  --green-3: #085a37;
  --red: #c73b3b;
}

.stApp {
  background:
    radial-gradient(1000px 560px at 10% -10%, #31385c 0%, transparent 65%),
    radial-gradient(900px 560px at 120% 20%, #2a2f50 0%, transparent 62%),
    linear-gradient(165deg, var(--bg-1) 0%, var(--bg-2) 56%, var(--bg-3) 100%);
  color: var(--ink);
  font-family: "Rajdhani", sans-serif;
}

[data-testid="stSidebar"] {
  display: none !important;
}

[data-testid="collapsedControl"] {
  display: none !important;
}

.arena-hero {
  border: 1px solid #2f426c;
  border-radius: 12px;
  padding: 14px 18px;
  background: linear-gradient(180deg, rgba(25, 33, 62, 0.92) 0%, rgba(17, 22, 43, 0.92) 100%);
  box-shadow: 0 10px 28px rgba(2, 6, 18, 0.45);
  margin-bottom: 0.7rem;
}

.control-panel {
  border: 1px solid #30486f;
  border-radius: 12px;
  background: linear-gradient(180deg, rgba(19, 28, 49, 0.92) 0%, rgba(15, 22, 40, 0.92) 100%);
  padding: 12px 14px;
  margin-bottom: 0.85rem;
}

.arena-title {
  font-family: "Orbitron", sans-serif;
  font-size: 1.55rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  color: #f6fbff;
  line-height: 1.1;
}

.arena-sub {
  font-size: 0.95rem;
  color: var(--muted);
}

.section-badge {
  display: inline-block;
  border: 1px solid #405c86;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #d0def9;
  background: rgba(47, 67, 102, 0.45);
  margin: 0.2rem 0 0.45rem 0;
}

.event-card {
  border: 1px solid #486a98;
  border-radius: 12px;
  padding: 12px 14px;
  background: linear-gradient(180deg, #1a2b4d 0%, #13223f 100%);
  color: #edf6ff;
  box-shadow: 0 8px 18px rgba(4, 13, 26, 0.3);
}

.event-card .title {
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 0.76rem;
  color: #b6d4f2;
}

.event-card .body {
  font-size: 1.02rem;
  line-height: 1.35;
}

[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] [role="columnheader"] {
  white-space: nowrap;
}

[data-testid="stDataFrame"] {
  border: 1px solid #30486f;
  border-radius: 10px;
  overflow: hidden;
}

[data-testid="stDataFrame"] [role="grid"] {
  background: #10172d;
}

[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] [role="columnheader"] {
  white-space: nowrap;
}

.poker-stage {
  position: relative;
  width: 100%;
  min-height: 510px;
  border: 1px solid #304b76;
  border-radius: 16px;
  margin: 0.4rem 0 1rem 0;
  background:
    radial-gradient(130% 90% at 50% 20%, rgba(60, 68, 106, 0.32) 0%, rgba(15, 19, 40, 0.18) 55%, rgba(11, 15, 31, 0.88) 100%);
  overflow: hidden;
}

.poker-table {
  position: absolute;
  left: 50%;
  top: 50%;
  width: min(83%, 760px);
  height: min(60%, 320px);
  transform: translate(-50%, -50%);
  border-radius: 50% / 44%;
  background: radial-gradient(ellipse at center, var(--green-1) 0%, var(--green-2) 60%, var(--green-3) 100%);
  border: 10px solid #1f2430;
  box-shadow:
    inset 0 0 28px rgba(0, 0, 0, 0.42),
    0 18px 34px rgba(0, 0, 0, 0.45);
}

.table-brand {
  position: absolute;
  left: 50%;
  top: 44%;
  transform: translate(-50%, -50%);
  opacity: 0.18;
  font-family: "Orbitron", sans-serif;
  font-size: clamp(1.4rem, 2.8vw, 2.1rem);
  letter-spacing: 0.08em;
}


.seat {
  position: absolute;
  width: 145px;
  padding: 7px 8px;
  border: 1px solid #4d617f;
  border-radius: 12px;
  background: linear-gradient(180deg, #161d2f 0%, #111727 100%);
  text-align: center;
  box-shadow: 0 7px 14px rgba(0, 0, 0, 0.45);
}

.seat .name {
  font-size: 0.96rem;
  font-weight: 700;
}

.seat .meta {
  font-size: 0.83rem;
  color: #9dd6b7;
}

.seat.elim .meta {
  color: #ff8f8f;
}

.seat.tag {
  border-color: #73a7dd;
}

.event-banner {
  position: absolute;
  left: 50%;
  top: 14px;
  transform: translateX(-50%);
  background: rgba(5, 12, 23, 0.82);
  border: 1px solid #4f6f98;
  border-radius: 999px;
  padding: 6px 14px;
  font-size: 0.9rem;
  max-width: min(90%, 780px);
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
}

.action-ribbon {
  margin-top: 8px;
  display: flex;
  gap: 8px;
}

.action-pill {
  flex: 1;
  padding: 10px 8px;
  text-align: center;
  border-radius: 10px;
  border: 1px solid #7c1f1f;
  background: linear-gradient(180deg, #c74141 0%, #9c2f2f 100%);
  color: #fff;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-size: 0.85rem;
}

.seat-0 { left: 50%; top: 22px; transform: translateX(-50%); }
.seat-1 { right: 54px; top: 122px; }
.seat-2 { right: 84px; bottom: 132px; }
.seat-3 { left: 50%; bottom: 18px; transform: translateX(-50%); }
.seat-4 { left: 84px; bottom: 132px; }
.seat-5 { left: 54px; top: 122px; }

@media (max-width: 980px) {
  .poker-stage {
    min-height: 620px;
  }
  .poker-table {
    width: min(92%, 650px);
    height: 260px;
    top: 48%;
  }
  .seat {
    width: 128px;
  }
  .seat-0 { top: 58px; }
  .seat-1 { right: 16px; top: 162px; }
  .seat-2 { right: 16px; bottom: 144px; }
  .seat-3 { bottom: 22px; }
  .seat-4 { left: 16px; bottom: 144px; }
  .seat-5 { left: 16px; top: 162px; }
}

[data-testid="stButton"] button {
  border-radius: 10px;
  border: 1px solid #415e8a;
  background: linear-gradient(180deg, #1a2e50 0%, #132342 100%);
  color: #eef7ff;
  font-weight: 600;
}

[data-testid="stButton"] button:hover {
  border-color: #5f86b8;
  color: #ffffff;
}

h1, h2, h3 {
  font-family: "Orbitron", sans-serif;
  letter-spacing: 0.03em;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_page_header():
    st.markdown(
        """
        <div class="arena-hero">
        <div class="arena-title">Coup Table View</div>
        <div class="arena-sub">Poker-style replay surface with advisor and belief overlays</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_badge(label):
    st.markdown(f'<span class="section-badge">{label}</span>', unsafe_allow_html=True)


_inject_theme()
_render_page_header()


if "events_raw" not in st.session_state:
    st.session_state.events_raw = []
if "table_player_count" not in st.session_state:
    st.session_state.table_player_count = 4
if "table_players" not in st.session_state:
    st.session_state.table_players = _default_player_names(st.session_state.table_player_count)
if "main_replay_index" not in st.session_state:
    st.session_state.main_replay_index = 0
if "main_play" not in st.session_state:
    st.session_state.main_play = False
if "main_speed" not in st.session_state:
    st.session_state.main_speed = 1.0
if "sim_results" not in st.session_state:
    st.session_state.sim_results = []
if "sim_replay_index" not in st.session_state:
    st.session_state.sim_replay_index = 0
if "sim_play" not in st.session_state:
    st.session_state.sim_play = False
if "sim_speed" not in st.session_state:
    st.session_state.sim_speed = 1.0
if "strict_no_duplicate_hand" not in st.session_state:
    st.session_state.strict_no_duplicate_hand = False
if "advisor_style" not in st.session_state:
    st.session_state.advisor_style = "Balanced"


def _render_top_controls():
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    _render_section_badge("Table Controls")

    c1, c2, c3, c4, c5, c6 = st.columns([1.5, 0.8, 1.4, 1.5, 1.0, 1.0])
    with c1:
        st.session_state.strict_no_duplicate_hand = st.toggle(
            "Strict no-duplicate-hand",
            value=bool(st.session_state.strict_no_duplicate_hand),
            key="strict_mode_toggle",
        )

    with c2:
        chosen_players = st.number_input(
            "Players",
            min_value=2,
            max_value=6,
            value=int(st.session_state.table_player_count),
            step=1,
            key="table_player_count_input",
        )
        if int(chosen_players) != int(st.session_state.table_player_count):
            st.session_state.table_player_count = int(chosen_players)
            st.session_state.table_players = _default_player_names(int(chosen_players))
            st.session_state.events_raw = []
            st.session_state.main_replay_index = 0
            st.session_state.main_play = False
            st.success(f"Started a new {int(chosen_players)}-player table.")

    with c3:
        styles = ["Conservative", "Balanced", "Aggressive"]
        current_style = str(st.session_state.get("advisor_style", "Balanced"))
        if current_style not in styles:
            current_style = "Balanced"
        st.session_state.advisor_style = st.radio(
            "Advisor Style",
            options=styles,
            index=styles.index(current_style),
            horizontal=True,
            key="advisor_style_radio",
        )

    with c4:
        uploaded = st.file_uploader(
            "Upload JSON events",
            type=["json"],
            key="events_upload_main",
            label_visibility="visible",
        )
        if uploaded is not None:
            try:
                payload = json.loads(uploaded.read().decode("utf-8"))
                events = parse_events(payload)
                st.session_state.events_raw = payload
                inferred = infer_players(events)
                if inferred:
                    roster = list(inferred)[:6]
                    st.session_state.table_players = roster
                    st.session_state.table_player_count = max(2, min(6, len(roster)))
                st.session_state.main_replay_index = 0
                st.success("Loaded event JSON")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid JSON events: {exc}")

    with c5:
        if st.button("Load sample game", use_container_width=True, key="load_sample_btn"):
            if SAMPLE_EVENTS_PATH.exists():
                payload = json.loads(SAMPLE_EVENTS_PATH.read_text())
                events = parse_events(payload)
                st.session_state.events_raw = payload
                inferred = infer_players(events)
                if inferred:
                    roster = list(inferred)[:6]
                    st.session_state.table_players = roster
                    st.session_state.table_player_count = max(2, min(6, len(roster)))
                st.session_state.main_replay_index = 0
                st.success("Loaded sample game")
            else:
                st.error(f"Sample file not found: {SAMPLE_EVENTS_PATH}")

    with c6:
        if st.button("Clear events", use_container_width=True, key="clear_events_btn"):
            st.session_state.events_raw = []
            st.session_state.main_replay_index = 0
            st.session_state.main_play = False

    st.markdown("</div>", unsafe_allow_html=True)


def _build_engine_from_raw(events_raw, *, strict_no_duplicate_hand):
    events = parse_events(events_raw)
    players = list(_configured_players())
    inferred = infer_players(events)
    for name in inferred:
        if name not in players:
            players.append(name)
    if not players:
        players = _default_player_names(4)

    engine = GameEngine(players, strict_no_duplicate_hand=strict_no_duplicate_hand)
    engine.replay(events)
    return engine, events


def _render_turn_builder_simple(strict_no_duplicate_hand):
    events_raw = st.session_state.events_raw
    role_values = [role.value for role in Role]
    target_actions = {"Steal", "Assassinate", "Coup"}

    try:
        engine, parsed_events = _build_engine_from_raw(
            events_raw,
            strict_no_duplicate_hand=strict_no_duplicate_hand,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot build turn controls because current events are invalid: {exc}")
        return

    players = sorted(engine.public_state.players.keys())
    alive_players = [
        name for name in players if int(engine.public_state.players[name].influence_alive) > 0
    ]
    if len(alive_players) < 2:
        st.info("Need at least 2 alive players to continue.")
        return

    st.caption("Choose actor and action, then apply turn. Targeted actions always require a target.")

    actor = st.selectbox("Actor", alive_players, key="simple_actor")
    actor_state = engine.public_state.players[actor]
    actor_coins = int(actor_state.coins)
    opponents = [name for name in alive_players if name != actor]

    legal_actions = ["Income", "Foreign Aid", "Tax", "Steal", "Exchange"]
    if actor_coins >= 3:
        legal_actions.insert(4, "Assassinate")
    if actor_coins >= 7:
        legal_actions.insert(5, "Coup")

    if actor_coins >= 10:
        legal_actions = ["Coup"]
        st.caption(f"{actor} has ${actor_coins}. Coup is mandatory at 10+ coins.")
    else:
        st.caption(f"{actor}: ${actor_coins}, {int(actor_state.influence_alive)} influence.")

    action_name = st.selectbox("Action", legal_actions, key="simple_action")
    target = None
    if action_name in target_actions:
        if not opponents:
            st.warning("No legal target is available.")
        else:
            target = st.selectbox("Target player", opponents, key="simple_target")

    claim_role = action_claim_role(action_name)
    action_challenge_mode = "none"
    action_challenger = None
    action_claim_outcome = "true"
    action_challenge_reveal = role_values[0]
    if claim_role is not None and opponents:
        action_challenge_mode = st.radio(
            f"Challenge {actor}'s {claim_role.value} claim?",
            options=["none", "challenge"],
            format_func=lambda item: "No challenge" if item == "none" else "Yes, challenge",
            horizontal=True,
            key="simple_action_challenge_mode",
        )
        if action_challenge_mode == "challenge":
            action_challenger = st.selectbox("Challenger", opponents, key="simple_action_challenger")
            action_claim_outcome = st.radio(
                "Challenge result",
                options=["true", "false"],
                format_func=lambda item: (
                    "Claim is true (challenger loses influence)"
                    if item == "true"
                    else "Claim is false (actor loses influence)"
                ),
                key="simple_action_claim_outcome",
            )
            action_challenge_loser = action_challenger if action_claim_outcome == "true" else actor
            action_challenge_reveal = st.selectbox(
                f"Revealed role for {action_challenge_loser}",
                role_values,
                key="simple_action_challenge_reveal",
            )

    foreign_aid_mode = "none"
    foreign_aid_blocker = None
    foreign_aid_actor_response = "accept"
    foreign_aid_outcome = "true"
    foreign_aid_reveal = role_values[0]
    if action_name == "Foreign Aid" and opponents:
        foreign_aid_mode = st.radio(
            "Foreign Aid response",
            options=["none", "block"],
            format_func=lambda item: (
                "No block (actor gains +2 coins)"
                if item == "none"
                else "Block with Duke claim"
            ),
            key="simple_fa_mode",
        )
        if foreign_aid_mode == "block":
            foreign_aid_blocker = st.selectbox("Blocker", opponents, key="simple_fa_blocker")
            foreign_aid_actor_response = st.radio(
                "Actor response to block",
                options=["accept", "challenge"],
                format_func=lambda item: (
                    "Accept block"
                    if item == "accept"
                    else "Challenge Duke block"
                ),
                key="simple_fa_actor_response",
            )
            if foreign_aid_actor_response == "challenge":
                foreign_aid_outcome = st.radio(
                    "Duke block challenge result",
                    options=["true", "false"],
                    format_func=lambda item: (
                        "Block claim is true (actor loses influence)"
                        if item == "true"
                        else "Block claim is false (blocker loses influence)"
                    ),
                    key="simple_fa_outcome",
                )
                fa_loser = actor if foreign_aid_outcome == "true" else foreign_aid_blocker
                foreign_aid_reveal = st.selectbox(
                    f"Revealed role for {fa_loser}",
                    role_values,
                    key="simple_fa_reveal",
                )

    steal_response = "proceed"
    steal_block_role = Role.CAPTAIN.value
    steal_actor_response = "accept"
    steal_block_outcome = "true"
    steal_block_reveal = role_values[0]
    if action_name == "Steal" and target is not None:
        target_state = engine.public_state.players[target]
        st.caption(f"{target}: ${int(target_state.coins)}, {int(target_state.influence_alive)} influence.")
        steal_response = st.radio(
            f"{target}'s response",
            options=["proceed", "block_captain", "block_ambassador"],
            format_func=lambda item: {
                "proceed": "Proceed (no block)",
                "block_captain": "Block with Captain",
                "block_ambassador": "Block with Ambassador",
            }[item],
            key="simple_steal_response",
        )
        if steal_response != "proceed":
            steal_block_role = (
                Role.CAPTAIN.value if steal_response == "block_captain" else Role.AMBASSADOR.value
            )
            steal_actor_response = st.radio(
                f"{actor}'s response to block",
                options=["accept", "challenge"],
                format_func=lambda item: (
                    "Accept block (Steal fails)"
                    if item == "accept"
                    else "Challenge block claim"
                ),
                key="simple_steal_actor_response",
            )
            if steal_actor_response == "challenge":
                steal_block_outcome = st.radio(
                    "Steal block challenge result",
                    options=["true", "false"],
                    format_func=lambda item: (
                        "Block claim is true (actor loses influence)"
                        if item == "true"
                        else "Block claim is false (blocker loses influence)"
                    ),
                    key="simple_steal_outcome",
                )
                steal_loser = actor if steal_block_outcome == "true" else target
                steal_block_reveal = st.selectbox(
                    f"Revealed role for {steal_loser}",
                    role_values,
                    key="simple_steal_reveal",
                )

    assassinate_response = "lose_influence"
    assassinate_target_reveal = role_values[0]
    assassinate_actor_response = "accept"
    assassinate_block_outcome = "true"
    assassinate_block_reveal = role_values[0]
    assassinate_followup_reveal = role_values[0]
    if action_name == "Assassinate" and target is not None:
        target_state = engine.public_state.players[target]
        st.caption(f"{target}: ${int(target_state.coins)}, {int(target_state.influence_alive)} influence.")
        assassinate_response = st.radio(
            f"{target}'s response",
            options=["lose_influence", "block"],
            format_func=lambda item: (
                "Lose 1 influence"
                if item == "lose_influence"
                else "Block with Contessa claim"
            ),
            key="simple_ass_response",
        )
        if assassinate_response == "lose_influence":
            assassinate_target_reveal = st.selectbox(
                f"Revealed role for {target}",
                role_values,
                key="simple_ass_target_reveal",
            )
        else:
            assassinate_actor_response = st.radio(
                f"{actor}'s response to Contessa block",
                options=["accept", "challenge"],
                format_func=lambda item: (
                    "Accept block"
                    if item == "accept"
                    else "Challenge Contessa block"
                ),
                key="simple_ass_actor_response",
            )
            if assassinate_actor_response == "challenge":
                assassinate_block_outcome = st.radio(
                    "Contessa block challenge result",
                    options=["true", "false"],
                    format_func=lambda item: (
                        "Block claim is true (actor loses influence)"
                        if item == "true"
                        else "Block claim is false (target loses influence, then assassination continues)"
                    ),
                    key="simple_ass_outcome",
                )
                ass_loser = actor if assassinate_block_outcome == "true" else target
                assassinate_block_reveal = st.selectbox(
                    f"Revealed role for {ass_loser}",
                    role_values,
                    key="simple_ass_block_reveal",
                )
                if assassinate_block_outcome == "false" and int(target_state.influence_alive) >= 2:
                    assassinate_followup_reveal = st.selectbox(
                        f"Revealed role for {target} (successful assassination)",
                        role_values,
                        key="simple_ass_followup_reveal",
                    )

    coup_reveal = role_values[0]
    if action_name == "Coup" and target is not None:
        coup_reveal = st.selectbox(
            f"Revealed role for {target}",
            role_values,
            key="simple_coup_reveal",
        )

    if st.button("Apply Turn", type="primary", key="simple_apply_turn"):
        errors = []
        events_to_add = []
        coins = {name: int(state.coins) for name, state in engine.public_state.players.items()}
        influence = {
            name: int(state.influence_alive) for name, state in engine.public_state.players.items()
        }

        def add_coin_change(player_name, delta):
            delta = int(delta)
            if delta == 0:
                return
            events_to_add.append(
                {
                    "type": "coin_change",
                    "player": player_name,
                    "delta": delta,
                }
            )
            coins[player_name] = max(0, int(coins.get(player_name, 0)) + delta)

        def lose_influence(player_name, revealed_role):
            if int(influence.get(player_name, 0)) <= 0:
                errors.append(f"{player_name} has no influence left to lose.")
                return
            events_to_add.append(
                {
                    "type": "reveal",
                    "player": player_name,
                    "revealed_role": revealed_role,
                }
            )
            influence[player_name] = max(0, int(influence.get(player_name, 0)) - 1)

        def resolve_claim_challenge(*, challenger, challenged, claimed_role, outcome, reveal_role):
            events_to_add.append(
                {
                    "type": "challenge",
                    "challenger": challenger,
                    "challenged": challenged,
                    "claimed_role": claimed_role,
                    "result": "win" if outcome == "true" else "lose",
                }
            )
            if outcome == "true":
                lose_influence(challenger, reveal_role)
                return True
            lose_influence(challenged, reveal_role)
            return False

        if actor not in alive_players:
            errors.append("Actor must be an alive player.")
        if action_name in target_actions and target is None:
            errors.append(f"{action_name} requires a target player.")
        if int(coins.get(actor, 0)) >= 10 and action_name != "Coup":
            errors.append(f"{actor} has 10+ coins and must Coup.")
        if action_name == "Coup" and int(coins.get(actor, 0)) < 7:
            errors.append("Coup requires at least 7 coins.")
        if action_name == "Assassinate" and int(coins.get(actor, 0)) < 3:
            errors.append("Assassinate requires at least 3 coins.")

        if errors:
            for message in errors:
                st.error(message)
            return

        events_to_add.append(
            {
                "type": "action",
                "actor": actor,
                "action_name": action_name,
                "target": target if action_name in target_actions else None,
            }
        )

        if action_name == "Coup":
            add_coin_change(actor, -7)
        elif action_name == "Assassinate":
            add_coin_change(actor, -3)

        action_continues = True
        if claim_role is not None and action_challenge_mode == "challenge" and action_challenger:
            action_continues = resolve_claim_challenge(
                challenger=action_challenger,
                challenged=actor,
                claimed_role=claim_role.value,
                outcome=action_claim_outcome,
                reveal_role=action_challenge_reveal,
            )
            if not action_continues and action_name == "Assassinate":
                add_coin_change(actor, 3)

        if action_name == "Income":
            add_coin_change(actor, 1)

        elif action_name == "Foreign Aid":
            if foreign_aid_mode == "none":
                add_coin_change(actor, 2)
            else:
                events_to_add.append(
                    {
                        "type": "block",
                        "blocker": foreign_aid_blocker,
                        "blocked_action": "Foreign Aid",
                        "roles_claimed": [Role.DUKE.value],
                    }
                )
                if foreign_aid_actor_response == "challenge":
                    block_holds = resolve_claim_challenge(
                        challenger=actor,
                        challenged=foreign_aid_blocker,
                        claimed_role=Role.DUKE.value,
                        outcome=foreign_aid_outcome,
                        reveal_role=foreign_aid_reveal,
                    )
                    if not block_holds:
                        add_coin_change(actor, 2)

        elif action_name == "Tax":
            if action_continues:
                add_coin_change(actor, 3)

        elif action_name == "Steal":
            if action_continues and target is not None and int(influence.get(target, 0)) > 0:
                if steal_response != "proceed":
                    events_to_add.append(
                        {
                            "type": "block",
                            "blocker": target,
                            "blocked_action": "Steal",
                            "roles_claimed": [steal_block_role],
                        }
                    )
                    block_holds = True
                    if steal_actor_response == "challenge":
                        block_holds = resolve_claim_challenge(
                            challenger=actor,
                            challenged=target,
                            claimed_role=steal_block_role,
                            outcome=steal_block_outcome,
                            reveal_role=steal_block_reveal,
                        )
                    if not block_holds and int(influence.get(target, 0)) > 0:
                        amount = min(2, int(coins.get(target, 0)))
                        add_coin_change(target, -amount)
                        add_coin_change(actor, amount)
                else:
                    amount = min(2, int(coins.get(target, 0)))
                    add_coin_change(target, -amount)
                    add_coin_change(actor, amount)

        elif action_name == "Assassinate":
            if action_continues and target is not None and int(influence.get(target, 0)) > 0:
                if assassinate_response == "block":
                    events_to_add.append(
                        {
                            "type": "block",
                            "blocker": target,
                            "blocked_action": "Assassination",
                            "roles_claimed": [Role.CONTESSA.value],
                        }
                    )
                    block_holds = True
                    if assassinate_actor_response == "challenge":
                        block_holds = resolve_claim_challenge(
                            challenger=actor,
                            challenged=target,
                            claimed_role=Role.CONTESSA.value,
                            outcome=assassinate_block_outcome,
                            reveal_role=assassinate_block_reveal,
                        )
                    if not block_holds and int(influence.get(target, 0)) > 0:
                        lose_influence(target, assassinate_followup_reveal)
                else:
                    lose_influence(target, assassinate_target_reveal)

        elif action_name == "Coup":
            if target is not None and int(influence.get(target, 0)) > 0:
                lose_influence(target, coup_reveal)

        if errors:
            for message in errors:
                st.error(message)
            return

        try:
            parsed_new_events = parse_events(events_to_add)
            validation_engine = GameEngine(
                players,
                strict_no_duplicate_hand=strict_no_duplicate_hand,
            )
            validation_engine.replay(parsed_events)
            for event in parsed_new_events:
                event_errors = validate_event(validation_engine.public_state, event)
                if event_errors:
                    raise ValueError("; ".join(event_errors))
                validation_engine.apply_event(event)

            new_events_raw = list(events_raw) + [
                event.model_dump(mode="json") for event in parsed_new_events
            ]
            st.session_state.events_raw = new_events_raw
            st.session_state.main_replay_index = max(0, len(new_events_raw) - 1)
            st.success(f"Added {len(parsed_new_events)} events for this turn.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to apply turn: {exc}")


def _render_guided_turn_form(strict_no_duplicate_hand):
    events_raw = st.session_state.events_raw
    role_values = [role.value for role in Role]
    target_actions = {"Steal", "Assassinate", "Coup"}

    try:
        engine, parsed_events = _build_engine_from_raw(
            events_raw,
            strict_no_duplicate_hand=strict_no_duplicate_hand,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot build guided turn form because current events are invalid: {exc}")
        return

    players = sorted(engine.public_state.players.keys())
    alive_players = [
        name for name in players if int(engine.public_state.players[name].influence_alive) > 0
    ]
    if len(alive_players) < 2:
        st.info("Need at least 2 alive players to build the next turn.")
        return

    with st.form("guided_turn_form", clear_on_submit=False):
        actor = st.selectbox("Actor", alive_players, key="guided_actor")
        actor_state = engine.public_state.players[actor]
        opponents = [name for name in alive_players if name != actor]

        actor_coins = int(actor_state.coins)
        legal_actions = ["Income", "Foreign Aid", "Tax", "Steal", "Exchange"]
        if actor_coins >= 3:
            legal_actions.insert(4, "Assassinate")
        if actor_coins >= 7:
            legal_actions.insert(5, "Coup")

        if actor_coins >= 10:
            legal_actions = ["Coup"]
            st.caption(f"{actor} has ${actor_coins}. Coup is mandatory at 10+ coins.")
        else:
            st.caption(f"{actor}: ${actor_coins}, {int(actor_state.influence_alive)} influence.")

        action_name = st.selectbox("Action", legal_actions, key="guided_action")
        target = None
        if action_name in target_actions:
            if not opponents:
                st.warning("No legal target is available.")
            else:
                target = st.selectbox("Target player", opponents, key="guided_target")

        claim_role = action_claim_role(action_name)
        action_challenge_mode = "none"
        action_challenger = None
        action_claim_outcome = "true"
        action_challenge_reveal = role_values[0]
        if claim_role is not None and opponents:
            action_challenge_mode = st.radio(
                f"Challenge {actor}'s {claim_role.value} claim?",
                options=["none", "challenge"],
                format_func=lambda item: "No challenge" if item == "none" else "Challenge claim",
                horizontal=True,
                key="guided_action_challenge_mode",
            )
            if action_challenge_mode == "challenge":
                action_challenger = st.selectbox("Challenger", opponents, key="guided_action_challenger")
                action_claim_outcome = st.radio(
                    "Action claim outcome",
                    options=["true", "false"],
                    format_func=lambda item: (
                        "Claim is true (challenger loses influence)"
                        if item == "true"
                        else "Claim is false (actor loses influence)"
                    ),
                    key="guided_action_claim_outcome",
                )
                action_challenge_loser = action_challenger if action_claim_outcome == "true" else actor
                action_challenge_reveal = st.selectbox(
                    f"Revealed role for {action_challenge_loser}",
                    role_values,
                    key="guided_action_challenge_reveal",
                )

        foreign_aid_mode = "none"
        foreign_aid_blocker = None
        foreign_aid_actor_response = "accept"
        foreign_aid_outcome = "true"
        foreign_aid_reveal = role_values[0]
        if action_name == "Foreign Aid" and opponents:
            foreign_aid_mode = st.radio(
                "Foreign Aid response",
                options=["none", "block"],
                format_func=lambda item: (
                    "No block (actor gains +2 coins)"
                    if item == "none"
                    else "Block with Duke claim"
                ),
                key="guided_fa_mode",
            )
            if foreign_aid_mode == "block":
                foreign_aid_blocker = st.selectbox(
                    "Blocker",
                    opponents,
                    key="guided_fa_blocker",
                )
                foreign_aid_actor_response = st.radio(
                    "Actor response to block",
                    options=["accept", "challenge"],
                    format_func=lambda item: (
                        "Accept block (Foreign Aid fails)"
                        if item == "accept"
                        else "Challenge Duke block"
                    ),
                    key="guided_fa_actor_response",
                )
                if foreign_aid_actor_response == "challenge":
                    foreign_aid_outcome = st.radio(
                        "Duke block outcome",
                        options=["true", "false"],
                        format_func=lambda item: (
                            "Block claim is true (actor loses influence)"
                            if item == "true"
                            else "Block claim is false (blocker loses influence)"
                        ),
                        key="guided_fa_block_outcome",
                    )
                    fa_loser = actor if foreign_aid_outcome == "true" else foreign_aid_blocker
                    foreign_aid_reveal = st.selectbox(
                        f"Revealed role for {fa_loser}",
                        role_values,
                        key="guided_fa_block_reveal",
                    )

        steal_response = "proceed"
        steal_block_role = Role.CAPTAIN.value
        steal_actor_response = "accept"
        steal_block_outcome = "true"
        steal_block_reveal = role_values[0]
        if action_name == "Steal" and target is not None:
            target_state = engine.public_state.players[target]
            st.caption(f"{target}: ${int(target_state.coins)}, {int(target_state.influence_alive)} influence.")
            steal_response = st.radio(
                f"{target}'s response",
                options=["proceed", "block"],
                format_func=lambda item: (
                    "Proceed (no block)"
                    if item == "proceed"
                    else "Block Steal (Captain or Ambassador)"
                ),
                key="guided_steal_response",
            )
            if steal_response == "block":
                steal_block_role = st.selectbox(
                    "Steal block role",
                    [Role.CAPTAIN.value, Role.AMBASSADOR.value],
                    key="guided_steal_block_role",
                )
                steal_actor_response = st.radio(
                    f"{actor}'s response to block",
                    options=["accept", "challenge"],
                    format_func=lambda item: (
                        "Accept block (Steal fails)"
                        if item == "accept"
                        else "Challenge block claim"
                    ),
                    key="guided_steal_actor_response",
                )
                if steal_actor_response == "challenge":
                    steal_block_outcome = st.radio(
                        "Steal block outcome",
                        options=["true", "false"],
                        format_func=lambda item: (
                            "Block claim is true (actor loses influence)"
                            if item == "true"
                            else "Block claim is false (blocker loses influence)"
                        ),
                        key="guided_steal_block_outcome",
                    )
                    steal_loser = actor if steal_block_outcome == "true" else target
                    steal_block_reveal = st.selectbox(
                        f"Revealed role for {steal_loser}",
                        role_values,
                        key="guided_steal_block_reveal",
                    )

        assassinate_response = "lose_influence"
        assassinate_target_reveal = role_values[0]
        assassinate_actor_response = "accept"
        assassinate_block_outcome = "true"
        assassinate_block_reveal = role_values[0]
        assassinate_followup_reveal = role_values[0]
        if action_name == "Assassinate" and target is not None:
            target_state = engine.public_state.players[target]
            st.caption(f"{target}: ${int(target_state.coins)}, {int(target_state.influence_alive)} influence.")
            assassinate_response = st.radio(
                f"{target}'s response",
                options=["lose_influence", "block"],
                format_func=lambda item: (
                    "Lose 1 influence"
                    if item == "lose_influence"
                    else "Block with Contessa claim"
                ),
                key="guided_ass_response",
            )
            if assassinate_response == "lose_influence":
                assassinate_target_reveal = st.selectbox(
                    f"Revealed role for {target}",
                    role_values,
                    key="guided_ass_target_reveal",
                )
            else:
                assassinate_actor_response = st.radio(
                    f"{actor}'s response to Contessa block",
                    options=["accept", "challenge"],
                    format_func=lambda item: (
                        "Accept block (Assassination fails)"
                        if item == "accept"
                        else "Challenge Contessa block"
                    ),
                    key="guided_ass_actor_response",
                )
                if assassinate_actor_response == "challenge":
                    assassinate_block_outcome = st.radio(
                        "Contessa block outcome",
                        options=["true", "false"],
                        format_func=lambda item: (
                            "Block claim is true (actor loses influence)"
                            if item == "true"
                            else "Block claim is false (target loses influence, then assassination continues)"
                        ),
                        key="guided_ass_block_outcome",
                    )
                    ass_loser = actor if assassinate_block_outcome == "true" else target
                    assassinate_block_reveal = st.selectbox(
                        f"Revealed role for {ass_loser}",
                        role_values,
                        key="guided_ass_block_reveal",
                    )
                    if assassinate_block_outcome == "false" and int(target_state.influence_alive) >= 2:
                        assassinate_followup_reveal = st.selectbox(
                            f"Revealed role for {target} (successful assassination)",
                            role_values,
                            key="guided_ass_followup_reveal",
                        )

        coup_reveal = role_values[0]
        if action_name == "Coup" and target is not None:
            coup_reveal = st.selectbox(
                f"Revealed role for {target}",
                role_values,
                key="guided_coup_reveal",
            )

        submitted = st.form_submit_button("Apply guided turn")
        if submitted:
            errors = []
            events_to_add = []
            coins = {name: int(state.coins) for name, state in engine.public_state.players.items()}
            influence = {
                name: int(state.influence_alive) for name, state in engine.public_state.players.items()
            }

            def add_coin_change(player_name, delta):
                delta = int(delta)
                if delta == 0:
                    return
                events_to_add.append(
                    {
                        "type": "coin_change",
                        "player": player_name,
                        "delta": delta,
                    }
                )
                coins[player_name] = max(0, int(coins.get(player_name, 0)) + delta)

            def lose_influence(player_name, revealed_role):
                if int(influence.get(player_name, 0)) <= 0:
                    errors.append(f"{player_name} has no influence left to lose.")
                    return
                events_to_add.append(
                    {
                        "type": "reveal",
                        "player": player_name,
                        "revealed_role": revealed_role,
                    }
                )
                influence[player_name] = max(0, int(influence.get(player_name, 0)) - 1)

            def resolve_claim_challenge(*, challenger, challenged, claimed_role, outcome, reveal_role):
                events_to_add.append(
                    {
                        "type": "challenge",
                        "challenger": challenger,
                        "challenged": challenged,
                        "claimed_role": claimed_role,
                        "result": "win" if outcome == "true" else "lose",
                    }
                )
                if outcome == "true":
                    lose_influence(challenger, reveal_role)
                    return True
                lose_influence(challenged, reveal_role)
                return False

            if actor not in alive_players:
                errors.append("Actor must be an alive player.")
            if action_name in target_actions and target is None:
                errors.append(f"{action_name} requires a target player.")
            if int(coins.get(actor, 0)) >= 10 and action_name != "Coup":
                errors.append(f"{actor} has 10+ coins and must Coup.")
            if action_name == "Coup" and int(coins.get(actor, 0)) < 7:
                errors.append("Coup requires at least 7 coins.")
            if action_name == "Assassinate" and int(coins.get(actor, 0)) < 3:
                errors.append("Assassinate requires at least 3 coins.")

            if errors:
                for message in errors:
                    st.error(message)
                return

            events_to_add.append(
                {
                    "type": "action",
                    "actor": actor,
                    "action_name": action_name,
                    "target": target if action_name in target_actions else None,
                }
            )

            if action_name == "Coup":
                add_coin_change(actor, -7)
            elif action_name == "Assassinate":
                add_coin_change(actor, -3)

            action_continues = True
            if claim_role is not None and action_challenge_mode == "challenge" and action_challenger:
                action_continues = resolve_claim_challenge(
                    challenger=action_challenger,
                    challenged=actor,
                    claimed_role=claim_role.value,
                    outcome=action_claim_outcome,
                    reveal_role=action_challenge_reveal,
                )
                if not action_continues and action_name == "Assassinate":
                    add_coin_change(actor, 3)

            if action_name == "Income":
                add_coin_change(actor, 1)

            elif action_name == "Foreign Aid":
                if foreign_aid_mode == "none":
                    add_coin_change(actor, 2)
                else:
                    events_to_add.append(
                        {
                            "type": "block",
                            "blocker": foreign_aid_blocker,
                            "blocked_action": "Foreign Aid",
                            "roles_claimed": [Role.DUKE.value],
                        }
                    )
                    if foreign_aid_actor_response == "challenge":
                        block_holds = resolve_claim_challenge(
                            challenger=actor,
                            challenged=foreign_aid_blocker,
                            claimed_role=Role.DUKE.value,
                            outcome=foreign_aid_outcome,
                            reveal_role=foreign_aid_reveal,
                        )
                        if not block_holds:
                            add_coin_change(actor, 2)

            elif action_name == "Tax":
                if action_continues:
                    add_coin_change(actor, 3)

            elif action_name == "Steal":
                if action_continues and target is not None and int(influence.get(target, 0)) > 0:
                    if steal_response == "block":
                        events_to_add.append(
                            {
                                "type": "block",
                                "blocker": target,
                                "blocked_action": "Steal",
                                "roles_claimed": [steal_block_role],
                            }
                        )
                        block_holds = True
                        if steal_actor_response == "challenge":
                            block_holds = resolve_claim_challenge(
                                challenger=actor,
                                challenged=target,
                                claimed_role=steal_block_role,
                                outcome=steal_block_outcome,
                                reveal_role=steal_block_reveal,
                            )
                        if not block_holds and int(influence.get(target, 0)) > 0:
                            amount = min(2, int(coins.get(target, 0)))
                            add_coin_change(target, -amount)
                            add_coin_change(actor, amount)
                    else:
                        amount = min(2, int(coins.get(target, 0)))
                        add_coin_change(target, -amount)
                        add_coin_change(actor, amount)

            elif action_name == "Assassinate":
                if action_continues and target is not None and int(influence.get(target, 0)) > 0:
                    if assassinate_response == "block":
                        events_to_add.append(
                            {
                                "type": "block",
                                "blocker": target,
                                "blocked_action": "Assassination",
                                "roles_claimed": [Role.CONTESSA.value],
                            }
                        )
                        block_holds = True
                        if assassinate_actor_response == "challenge":
                            block_holds = resolve_claim_challenge(
                                challenger=actor,
                                challenged=target,
                                claimed_role=Role.CONTESSA.value,
                                outcome=assassinate_block_outcome,
                                reveal_role=assassinate_block_reveal,
                            )
                        if not block_holds and int(influence.get(target, 0)) > 0:
                            lose_influence(target, assassinate_followup_reveal)
                    else:
                        lose_influence(target, assassinate_target_reveal)

            elif action_name == "Coup":
                if target is not None and int(influence.get(target, 0)) > 0:
                    lose_influence(target, coup_reveal)

            if errors:
                for message in errors:
                    st.error(message)
                return

            try:
                parsed_new_events = parse_events(events_to_add)
                validation_engine = GameEngine(
                    players,
                    strict_no_duplicate_hand=strict_no_duplicate_hand,
                )
                validation_engine.replay(parsed_events)
                for event in parsed_new_events:
                    event_errors = validate_event(validation_engine.public_state, event)
                    if event_errors:
                        raise ValueError("; ".join(event_errors))
                    validation_engine.apply_event(event)

                new_events_raw = list(events_raw) + [
                    event.model_dump(mode="json") for event in parsed_new_events
                ]
                st.session_state.events_raw = new_events_raw
                st.session_state.main_replay_index = max(0, len(new_events_raw) - 1)
                st.success(f"Added {len(parsed_new_events)} events from guided turn.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to apply guided turn: {exc}")


def _render_add_event_form():
    events_raw = st.session_state.events_raw
    players = _candidate_players(events_raw)

    with st.form("add_event_form", clear_on_submit=False):
        event_type = st.selectbox(
            "Event type",
            ["action", "block", "challenge", "reveal", "coin_change", "influence_lost"],
        )

        payload = {"type": event_type}

        if event_type == "action":
            actor = st.selectbox("Actor", players, key="add_actor")
            action_name = st.selectbox("Action", ACTION_NAMES, key="add_action_name")
            payload["actor"] = actor
            payload["action_name"] = action_name

            target_options = [name for name in players if name != actor]
            if action_name in {"Steal", "Assassinate", "Coup"}:
                if target_options:
                    payload["target"] = st.selectbox(
                        "Target",
                        target_options,
                        key="add_action_target",
                    )
                else:
                    payload["target"] = None
                    st.warning("No valid target available.")
            else:
                payload["target"] = None

        elif event_type == "block":
            payload["blocker"] = st.selectbox("Blocker", players, key="add_blocker")
            payload["blocked_action"] = st.selectbox(
                "Blocked action",
                BLOCKED_ACTIONS,
                key="add_blocked_action",
            )
            payload["roles_claimed"] = st.multiselect(
                "Roles claimed",
                [role.value for role in Role],
                default=[Role.DUKE.value],
                key="add_roles_claimed",
            )

        elif event_type == "challenge":
            payload["challenger"] = st.selectbox("Challenger", players, key="add_challenger")
            payload["challenged"] = st.selectbox("Challenged", players, key="add_challenged")
            payload["claimed_role"] = st.selectbox(
                "Claimed role",
                [role.value for role in Role],
                key="add_claimed_role",
            )
            payload["result"] = st.selectbox("Result", ["win", "lose"], key="add_result")
            revealed = st.selectbox(
                "Revealed role (optional)",
                ["-"] + [role.value for role in Role],
                key="add_challenge_revealed_role",
            )
            payload["revealed_role"] = None if revealed == "-" else revealed

        elif event_type in {"reveal", "influence_lost"}:
            payload["player"] = st.selectbox("Player", players, key=f"add_player_{event_type}")
            payload["revealed_role"] = st.selectbox(
                "Revealed role",
                [role.value for role in Role],
                key=f"add_revealed_{event_type}",
            )

        elif event_type == "coin_change":
            payload["player"] = st.selectbox("Player", players, key="add_coin_player")
            payload["delta"] = int(st.number_input("Delta", value=0, step=1, key="add_coin_delta"))

        submitted = st.form_submit_button("Add raw event")
        if submitted:
            try:
                if event_type == "action" and payload["action_name"] in {"Steal", "Assassinate", "Coup"}:
                    if payload.get("target") is None:
                        raise ValueError(f"{payload['action_name']} requires a target player.")
                if event_type == "block" and not payload.get("roles_claimed"):
                    raise ValueError("Block event must include at least one claimed role.")

                new_events = list(events_raw) + [payload]
                parse_events(new_events)
                st.session_state.events_raw = new_events
                st.session_state.main_replay_index = max(0, len(new_events) - 1)
                st.success("Raw event added")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid raw event: {exc}")

def _render_belief_delta(engine_before, engine_after):
    """
    Display a compact table showing how each player's role probabilities
    changed between two engine states (before and after an event).
    Uses color-coded arrows: ↑ for increases, ↓ for decreases.
    """
    _render_section_badge("Belief Δ (Change from this event)")

    before_probs = engine_before.belief_state.probabilities
    after_probs = engine_after.belief_state.probabilities

    all_players = sorted(set(before_probs.keys()) | set(after_probs.keys()))
    roles = list(Role)

    delta_rows = []
    for player in all_players:
        after_state = engine_after.public_state.players.get(player)
        if after_state and int(after_state.influence_alive) <= 0:
            continue  # skip eliminated players

        row = {"Player": player}
        has_delta = False
        for role in roles:
            p_before = float(before_probs.get(player, {}).get(role, 0.0))
            p_after = float(after_probs.get(player, {}).get(role, 0.0))
            delta = p_after - p_before

            if abs(delta) < 0.005:
                row[role.value] = f"{p_after:.2f}"
            elif delta > 0:
                row[role.value] = f"{p_after:.2f} ↑{delta:+.2f}"
                has_delta = True
            else:
                row[role.value] = f"{p_after:.2f} ↓{delta:+.2f}"
                has_delta = True

        if has_delta:
            delta_rows.append(row)

    if not delta_rows:
        st.caption("No significant belief changes from this event.")
        return

    import pandas as pd

    df = pd.DataFrame(delta_rows).set_index("Player")
    st.dataframe(df, use_container_width=True)
    st.caption("↑ probability increased · ↓ probability decreased · values < 0.005 shown as stable")


def _render_replay(events_raw, *, strict_no_duplicate_hand, prefix):
    if not events_raw:
        _render_section_badge("Coup Table")
        engine_empty = GameEngine(["P1", "P2", "P3", "P4"], strict_no_duplicate_hand=strict_no_duplicate_hand)
        _render_poker_table(
            engine_empty,
            current_event="Waiting for events...",
            event_index=0,
            total_events=0,
        )
        st.info("No events loaded. Use the controls above or Add Event section.")
        return

    try:
        events = parse_events(events_raw)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Invalid events payload: {exc}")
        return

    total = len(events)
    if total <= 0:
        _render_section_badge("Coup Table")
        players = _configured_players()
        if not players:
            players = _default_player_names(4)
        engine_empty = GameEngine(
            players,
            strict_no_duplicate_hand=strict_no_duplicate_hand,
        )
        _render_poker_table(
            engine_empty,
            current_event="No parsed events available.",
            event_index=0,
            total_events=0,
        )
        st.info("No parsed events available for replay.")
        return

    replay_key = f"{prefix}_replay_index"
    play_key = f"{prefix}_play"
    speed_key = f"{prefix}_speed"

    if st.session_state[replay_key] >= total:
        st.session_state[replay_key] = max(0, total - 1)

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### Controls")
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Jump to Start", key=f"{prefix}_start"):
            st.session_state[replay_key] = 0
            st.session_state[play_key] = False
        if c2.button("Prev", key=f"{prefix}_prev"):
            st.session_state[replay_key] = max(0, st.session_state[replay_key] - 1)
            st.session_state[play_key] = False
        if c3.button("Next", key=f"{prefix}_next"):
            st.session_state[replay_key] = min(total - 1, st.session_state[replay_key] + 1)
            st.session_state[play_key] = False
        if c4.button("Jump to End", key=f"{prefix}_end"):
            st.session_state[replay_key] = total - 1
            st.session_state[play_key] = False

        p1, p2 = st.columns(2)
        st.session_state[play_key] = p1.toggle("Play / Pause", value=st.session_state[play_key], key=f"{prefix}_play_toggle")
        st.session_state[speed_key] = p2.select_slider(
            "Speed",
            options=[0.25, 0.5, 1.0, 2.0, 4.0],
            value=float(st.session_state[speed_key]),
            format_func=lambda val: f"{val}x",
            key=f"{prefix}_speed_slider",
        )

        timeline_max = total - 1
        if timeline_max == 0:
            st.session_state[replay_key] = 0
            st.caption("Timeline has a single event; slider is disabled.")
        else:
            slider_index = st.slider(
                "Timeline",
                min_value=0,
                max_value=timeline_max,
                value=int(st.session_state[replay_key]),
                key=f"{prefix}_timeline",
            )
            st.session_state[replay_key] = int(slider_index)

    event_index = int(st.session_state[replay_key])
    current_event = events[event_index]

    engine_before = _build_engine(events, event_index - 1, strict_no_duplicate_hand)
    warnings = validate_event(engine_before.public_state, current_event)

    engine_after = _build_engine(events, event_index, strict_no_duplicate_hand)
    _render_section_badge("Coup Table")
    _render_poker_table(
        engine_after,
        current_event=_event_to_sentence(current_event),
        event_index=event_index + 1,
        total_events=total,
    )

    with right:
        _render_section_badge("Current Event")
        st.markdown(
            f"""
<div class="event-card">
  <div class="title">Live Action</div>
  <div class="body">{html.escape(_event_to_sentence(current_event))}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption(f"Event {event_index + 1}/{total}")

    detail_left, detail_right = st.columns([1.3, 1])
    advisor_style = str(st.session_state.get("advisor_style", "Balanced"))
    with detail_left:
        _render_section_badge("Advisor Desk")
        alive_after = [
            name
            for name, state in engine_after.public_state.players.items()
            if int(state.influence_alive) > 0
        ]
        perspective_key = f"{prefix}_advisor_perspective"
        if not alive_after:
            perspective = None
        else:
            default_perspective = st.session_state.get(perspective_key, alive_after[0])
            if default_perspective not in alive_after:
                default_perspective = alive_after[0]
            perspective = st.selectbox(
                "Perspective player",
                alive_after,
                index=alive_after.index(default_perspective),
                key=perspective_key,
            )

        st.caption(f"Advisor style: {advisor_style}")

        advisor_rows = _advisor_rows(engine_before, current_event, style=advisor_style)
        if advisor_rows:
            st.dataframe(pd.DataFrame(advisor_rows), use_container_width=True)
            for row in advisor_rows:
                st.caption(row["human_text"])
        else:
            st.write("No claim event at this step.")

        if perspective is not None:
            brief = _advisor_brief(engine_after, perspective=perspective, style=advisor_style)
            st.markdown("**Top Warnings**")
            for warning in brief["warnings"]:
                st.write(f"- {warning}")

            st.markdown("**Top 3 Paths**")
            if brief["options"]:
                for index, option in enumerate(brief["options"], start=1):
                    st.write(f"{index}. {option['title']}")
                    st.caption(option["detail"])
            else:
                st.write("No legal options available.")

            st.caption(brief["note"])

    with detail_right:
        _render_section_badge("Validation")
        if warnings:
            st.warning("Validation warnings")
            for warning in warnings:
                st.write(f"- {warning}")
        else:
            st.success("No validation warnings")
        st.divider()
        _render_belief_delta(engine_before, engine_after)

    _render_section_badge("Public State")
    st.dataframe(_public_state_table(engine_after), use_container_width=True)

    _render_section_badge("Revealed Dead")
    st.dataframe(_revealed_dead_table(engine_after), use_container_width=True)

    _render_section_badge("Belief Table")
    st.dataframe(_belief_table(engine_after), use_container_width=True)

    _render_section_badge("Event Log")
    event_rows = [_event_row(event) for event in events]
    st.dataframe(pd.DataFrame(event_rows), use_container_width=True, height=260)

    with st.expander("Debug: Raw Event JSON", expanded=False):
        st.json(current_event.model_dump(mode="json"))

    _maybe_autoplay(prefix, total)


def _render_simulator_controls(strict_no_duplicate_hand):
    _render_section_badge("Simulation Control Room")
    controls = st.columns(5)
    players = controls[0].number_input("Players", min_value=2, max_value=6, value=4, step=1)
    games = controls[1].number_input("Games", min_value=1, max_value=200, value=10, step=1)
    seed = controls[2].number_input("Seed", min_value=0, value=42, step=1)
    max_turns = controls[3].number_input("Max turns", min_value=20, max_value=500, value=200, step=10)
    bot_mix_input = controls[4].text_input("Bot mix (comma)", value="honest,bluffer,aggressive,cautious")

    if st.button("Run simulation batch"):
        bot_mix = [part.strip() for part in bot_mix_input.split(",") if part.strip()]
        results = simulate_games(
            int(games),
            players=int(players),
            seed=int(seed),
            bot_mix=bot_mix,
            max_turns=int(max_turns),
        )
        st.session_state.sim_results = [
            {
                "players": result.players,
                "winner": result.winner,
                "turns": result.turns,
                "events": result.events,
            }
            for result in results
        ]
        st.session_state.sim_replay_index = 0
        st.success(f"Generated {len(results)} games")

    if not st.session_state.sim_results:
        st.info("Run a simulation batch to inspect generated games.")
        return

    _render_section_badge("Simulation Summary")
    summary_rows = []
    for idx, result in enumerate(st.session_state.sim_results, start=1):
        summary_rows.append(
            {
                "game": idx,
                "winner": result["winner"],
                "turns": result["turns"],
                "events": len(result["events"]),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    game_idx = st.selectbox(
        "Select game",
        options=list(range(len(st.session_state.sim_results))),
        format_func=lambda idx: f"Game {idx + 1}",
        key="sim_selected_game",
    )
    selected = st.session_state.sim_results[int(game_idx)]

    left, right = st.columns(2)
    if left.button("Load selected game into Replay tab"):
        st.session_state.events_raw = list(selected["events"])
        if selected.get("players"):
            roster = list(selected["players"])[:6]
            st.session_state.table_players = roster
            st.session_state.table_player_count = max(2, min(6, len(roster)))
        st.session_state.main_replay_index = 0
        st.success("Loaded selected simulation into Replay tab")
    if right.button("Export selected game JSON"):
        output = Path("data") / "sim_example.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(selected["events"], indent=2))
        st.success(f"Saved: {output}")

    _render_section_badge("Simulated Match Replay")
    _render_replay(
        events_raw=selected["events"],
        strict_no_duplicate_hand=strict_no_duplicate_hand,
        prefix="sim",
    )


def _build_engine(events, end_index, strict_no_duplicate_hand):
    base_players = list(_configured_players())

    if end_index < 0:
        players = list(base_players)
        inferred = infer_players(events)
        for name in inferred:
            if name not in players:
                players.append(name)
        if not players:
            players = _default_player_names(4)
        return GameEngine(players, strict_no_duplicate_hand=strict_no_duplicate_hand)

    sub_events = events[: end_index + 1]
    players = list(base_players)
    inferred = infer_players(sub_events)
    for name in inferred:
        if name not in players:
            players.append(name)
    if not players:
        players = _default_player_names(4)
    engine = GameEngine(players, strict_no_duplicate_hand=strict_no_duplicate_hand)
    engine.replay(sub_events)
    return engine


def _clamp(value, low=0.0, high=1.0):
    value = float(value)
    if value < low:
        return float(low)
    if value > high:
        return float(high)
    return float(value)


def _style_risk_weight(style):
    name = str(style)
    if name == "Conservative":
        return 1.2
    if name == "Aggressive":
        return 0.8
    return 1.0


def _legal_actions_for_coins(coins):
    coins = int(coins)
    if coins >= 10:
        return ["Coup"]

    actions = ["Income", "Foreign Aid", "Tax", "Steal", "Exchange"]
    if coins >= 3:
        actions.insert(4, "Assassinate")
    if coins >= 7:
        actions.insert(5, "Coup")
    return actions


def _advisor_brief(engine, *, perspective, style):
    players = engine.public_state.players
    belief = engine.belief_state.probabilities
    style_risk = _style_risk_weight(style)

    if perspective not in players:
        return {
            "warnings": ["Selected perspective player is not present in this state."],
            "options": [],
            "note": "Belief values are role-likelihood scores from observed events, not exact hidden-card truth.",
        }

    me_state = players[perspective]
    if int(me_state.influence_alive) <= 0:
        return {
            "warnings": [f"{perspective} is eliminated and has no legal actions."],
            "options": [],
            "note": "Belief values are role-likelihood scores from observed events, not exact hidden-card truth.",
        }

    alive_opponents = [
        name
        for name, state in players.items()
        if name != perspective and int(state.influence_alive) > 0
    ]

    warnings = []
    for name in alive_opponents:
        state = players[name]
        probs = belief.get(name, {})
        coins = int(state.coins)

        if coins >= 7:
            warnings.append(f"High pressure: {name} has {coins} coins and is close to Coup range.")

        assassin_prob = float(probs.get(Role.ASSASSIN, 0.0))
        if coins >= 3 and assassin_prob >= 0.30:
            warnings.append(
                f"Danger: {name} can assassinate now (Assassin likelihood {assassin_prob:.2f})."
            )

        steal_prob = float(probs.get(Role.CAPTAIN, 0.0)) + float(probs.get(Role.AMBASSADOR, 0.0))
        if coins > 0 and steal_prob >= 0.45:
            warnings.append(
                f"Steal risk: {name} likely has Captain/Ambassador line ({steal_prob:.2f})."
            )

    tax_claimers = []
    for event in engine.public_state.history:
        if isinstance(event, ActionEvent) and event.action_name == "Tax":
            if event.actor not in tax_claimers:
                tax_claimers.append(event.actor)
    if len(tax_claimers) > 3:
        min_bluffs = len(tax_claimers) - 3
        warnings.append(
            f"Tax overload: {len(tax_claimers)} players claimed Duke; at least {min_bluffs} is bluffing."
        )

    if not warnings:
        warnings.append("Stable board now: monitor coin spikes and repeated role claims.")

    me_probs = belief.get(perspective, {})
    legal_actions = _legal_actions_for_coins(me_state.coins)

    def challenge_pressure(role_prob):
        role_prob = _clamp(role_prob)
        pressure = (1.0 - role_prob) * (0.45 + 0.08 * len(alive_opponents))
        return _clamp(pressure, 0.0, 0.95)

    def opponent_threat(name):
        state = players[name]
        coins = int(state.coins)
        influence = int(state.influence_alive)
        probs = belief.get(name, {})

        # Kill bonus: massive priority for finishing off a 1-influence player
        kill_bonus = 6.0 if influence == 1 else 0.0

        # Coin threat: urgency based on how close they are to Coup
        coin_threat = coins / 3.0

        # Role danger to ME specifically:
        # If they have Assassin + 3 coins, they can kill me next turn
        me_state = players.get(perspective)
        my_influence = int(me_state.influence_alive) if me_state else 2
        assassin_danger = 0.0
        if coins >= 3 and my_influence <= 1:
            assassin_danger = float(probs.get(Role.ASSASSIN, 0.0)) * 3.0

        # General board presence
        board_presence = float(influence)

        return kill_bonus + coin_threat + assassin_danger + board_presence

    ranked_targets = sorted(alive_opponents, key=opponent_threat, reverse=True)
    best_target = ranked_targets[0] if ranked_targets else None

    options = []

    income_score = 1.0
    options.append(
        {
            "title": "Income",
            "score": income_score,
            "detail": "If board is volatile, take +1 guaranteed coin. Else, use this to avoid risky claims.",
        }
    )

    if "Foreign Aid" in legal_actions:
        duke_block_risk = 0.0
        for name in alive_opponents:
            duke_block_risk = max(duke_block_risk, float(belief.get(name, {}).get(Role.DUKE, 0.0)))
        fa_score = 2.0 * (1.0 - style_risk * 0.55 * duke_block_risk)
        options.append(
            {
                "title": "Foreign Aid",
                "score": fa_score,
                "detail": (
                    f"If no Duke block appears, gain +2. If blocked, challenge only when Duke confidence is weak "
                    f"({style.lower()} mode)."
                ),
            }
        )

    if "Tax" in legal_actions:
        duke_truth = float(me_probs.get(Role.DUKE, 0.20))
        tax_challenge_risk = challenge_pressure(duke_truth)
        tax_score = 3.0 * (1.0 - style_risk * tax_challenge_risk)
        options.append(
            {
                "title": "Tax (Duke claim)",
                "score": tax_score,
                "detail": (
                    f"If table challenge pressure is low ({tax_challenge_risk:.2f}), claim +3 for tempo. "
                    f"If pressure rises, switch to safer lines."
                ),
            }
        )

    if "Steal" in legal_actions and best_target is not None:
        target_state = players[best_target]
        steal_amount = min(2, int(target_state.coins))
        target_probs = belief.get(best_target, {})
        block_risk = float(target_probs.get(Role.CAPTAIN, 0.0)) + float(
            target_probs.get(Role.AMBASSADOR, 0.0)
        )
        steal_truth = float(me_probs.get(Role.CAPTAIN, 0.20))
        steal_challenge_risk = challenge_pressure(steal_truth)
        steal_score = float(steal_amount) * (
            1.0 - style_risk * (0.65 * block_risk + 0.35 * steal_challenge_risk)
        )
        options.append(
            {
                "title": f"Steal from {best_target}",
                "score": steal_score,
                "detail": (
                    f"If not blocked, gain up to +{steal_amount}. If blocked, challenge only when "
                    f"Captain/Ambassador confidence is weak; otherwise accept block."
                ),
            }
        )

    if "Assassinate" in legal_actions and best_target is not None:
        target_state = players[best_target]
        target_probs = belief.get(best_target, {})
        block_contessa = float(target_probs.get(Role.CONTESSA, 0.0))
        ass_truth = float(me_probs.get(Role.ASSASSIN, 0.20))
        ass_challenge_risk = challenge_pressure(ass_truth)
        kill_value = 2.8 if int(target_state.influence_alive) == 1 else 1.8
        ass_score = kill_value * (
            1.0 - style_risk * (0.7 * block_contessa + 0.3 * ass_challenge_risk)
        ) - 0.4
        options.append(
            {
                "title": f"Assassinate {best_target}",
                "score": ass_score,
                "detail": (
                    f"If it lands, remove 1 influence. If Contessa block appears, challenge only with low "
                    f"Contessa confidence and favorable style."
                ),
            }
        )

    if "Coup" in legal_actions and best_target is not None:
        target_state = players[best_target]
        coup_value = 3.2 if int(target_state.influence_alive) == 1 else 2.4
        options.append(
            {
                "title": f"Coup {best_target}",
                "score": coup_value,
                "detail": "Guaranteed influence removal. Use when you need certainty over bluff-based lines.",
            }
        )

    if "Exchange" in legal_actions:
        pressure_bonus = 0.0
        for name in alive_opponents:
            pressure_bonus = max(pressure_bonus, float(players[name].coins) / 10.0)
        exchange_score = 1.1 + 0.6 * pressure_bonus
        options.append(
            {
                "title": "Exchange (Ambassador claim)",
                "score": exchange_score,
                "detail": "If your line looks exposed, exchange to reset hidden options and future bluff space.",
            }
        )

    options = sorted(options, key=lambda item: float(item["score"]), reverse=True)[:3]
    return {
        "warnings": warnings[:3],
        "options": options,
        "note": "Belief values are role-likelihood scores from observed events, not exact hidden-card truth.",
    }


def _advisor_rows(engine, event, *, style="Balanced"):
    claims = []
    if isinstance(event, ActionEvent):
        role = action_claim_role(event.action_name)
        if role is not None:
            claims.append(
                {
                    "claimant": event.actor,
                    "role": role,
                    "action_name": event.action_name,
                    "block_type": None,
                }
            )
    elif isinstance(event, BlockEvent):
        for role in block_claim_roles(event.blocked_action):
            claims.append(
                {
                    "claimant": event.blocker,
                    "role": role,
                    "action_name": None,
                    "block_type": event.blocked_action,
                }
            )

    if not claims:
        return []

    remaining = engine.remaining_copies()
    rows = []
    for claim in claims:
        context = ClaimContext(
            action_name=claim["action_name"],
            block_type=claim["block_type"],
            remaining_copies=remaining.get(claim["role"], 0),
        )
        rec = recommend_challenge(
            engine.belief_state,
            claim["claimant"],
            claim["role"],
            context,
            style=style,
        )

        if rec.recommendation == "Challenge":
            human_text = (
                f"{style} call: challenge {claim['claimant']}'s {claim['role'].value} claim. "
                f"If you are wrong, you lose influence; truth estimate {rec.p_truth:.2f} is below "
                f"threshold {rec.threshold:.2f}."
            )
        else:
            human_text = (
                f"{style} call: let {claim['claimant']}'s {claim['role'].value} claim pass. "
                f"Truth estimate {rec.p_truth:.2f} is above threshold {rec.threshold:.2f}; "
                "hold influence for next decision."
            )

        rows.append(
            {
                "claimant": claim["claimant"],
                "role": claim["role"].value,
                "recommendation": rec.recommendation,
                "p_truth": round(float(rec.p_truth), 3),
                "threshold": round(float(rec.threshold), 3),
                "human_text": human_text,
            }
        )

    return rows


def _public_state_table(engine):
    rows = []
    for name, state in engine.public_state.players.items():
        rows.append(
            {
                "player": name,
                "coins": int(state.coins),
                "influence": int(state.influence_alive),
            }
        )
    return pd.DataFrame(rows)


def _revealed_dead_table(engine):
    rows = []
    remaining = engine.remaining_copies()
    for role in Role:
        rows.append(
            {
                "role": role.value,
                "revealed": int(engine.public_state.revealed_dead.get(role, 0)),
                "remaining": int(remaining.get(role, 0)),
            }
        )
    return pd.DataFrame(rows)


def _belief_table(engine):
    rows = []
    for player, probs in engine.belief_state.probabilities.items():
        state = engine.public_state.players.get(player)
        if state and state.influence_alive <= 0:
            row = {"player": player}
            for role in Role:
                row[role.value] = "ELIM"
            rows.append(row)
            continue

        row = {"player": player}
        for role in Role:
            row[role.value] = round(float(probs.get(role, 0.0)), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _event_row(event):
    data = event.model_dump(mode="json")
    data["event"] = _event_to_sentence(event)
    return data


def _event_to_sentence(event):
    if isinstance(event, ActionEvent):
        target = f" -> {event.target}" if event.target else ""
        return f"{event.actor} uses {event.action_name}{target}"
    if isinstance(event, BlockEvent):
        claimed = ", ".join(role.value for role in event.roles_claimed)
        return f"{event.blocker} blocks {event.blocked_action} claiming {claimed}"

    payload = event.model_dump(mode="json")
    kind = payload.get("type", "event")
    detail = ", ".join(f"{key}={value}" for key, value in payload.items() if key != "type")
    return f"{kind}: {detail}"


def _candidate_players(events_raw):
    names = list(_configured_players())
    try:
        events = parse_events(events_raw)
        inferred = infer_players(events)
        for player_name in inferred:
            if player_name not in names:
                names.append(player_name)
    except Exception:  # noqa: BLE001
        pass

    if names:
        return names[:6]
    return _default_player_names(4)


def _render_poker_table(engine, *, current_event, event_index, total_events):
    players = sorted(engine.public_state.players.items(), key=lambda item: item[0])[:6]
    leader_name, _, _ = _leader_snapshot(engine)
    total_coins = sum(int(state.coins) for _, state in players)

    seats = []
    for idx, (name, state) in enumerate(players):
        influence = int(state.influence_alive)
        coins = int(state.coins)
        status = "ELIMINATED" if influence <= 0 else f"{influence} influence"

        classes = [f"seat seat-{idx}"]
        if influence <= 0:
            classes.append("elim")
        if name == leader_name:
            classes.append("tag")

        seat_html = f"""
<div class="{' '.join(classes)}">
  <div class="name">{html.escape(name)}</div>
  <div class="meta">${coins} • {html.escape(status)}</div>
</div>
"""
        seats.append(seat_html)

    if not seats:
        seats.append(
            """
<div class="seat seat-3">
  <div class="name">No players</div>
  <div class="meta">Load events to start replay</div>
</div>
"""
        )

    stage_html = f"""
<div class="poker-stage">
  <div class="event-banner">Event {event_index}/{total_events} • {html.escape(current_event)}</div>
  <div class="poker-table">
    <div class="table-brand">COUP TABLE</div>
  </div>
  {''.join(seats)}
</div>
<div class="action-ribbon">
  <div class="action-pill">Challenge</div>
  <div class="action-pill">Allow</div>
  <div class="action-pill">Coup Window</div>
</div>
"""
    st.markdown(stage_html, unsafe_allow_html=True)


def _leader_snapshot(engine):
    if not engine.public_state.players:
        return "-", 0, 0

    leader_name = "-"
    leader_coins = -1
    leader_influence = -1
    for name, state in engine.public_state.players.items():
        coins = int(state.coins)
        influence = int(state.influence_alive)
        if influence > leader_influence or (influence == leader_influence and coins > leader_coins):
            leader_name = name
            leader_coins = coins
            leader_influence = influence
    return leader_name, leader_coins, leader_influence


def _maybe_autoplay(prefix, total_events):
    play_key = f"{prefix}_play"
    speed_key = f"{prefix}_speed"
    index_key = f"{prefix}_replay_index"
    tick_key = f"{prefix}_last_tick"

    if not st.session_state.get(play_key):
        st.session_state[tick_key] = time.time()
        return

    interval = 1.0 / float(st.session_state[speed_key])
    now = time.time()
    last = float(st.session_state.get(tick_key, 0.0))
    if now - last < interval:
        return

    if st.session_state[index_key] >= total_events - 1:
        st.session_state[play_key] = False
        return

    st.session_state[index_key] += 1
    st.session_state[tick_key] = now
    st.rerun()


def _placeholder_frame():
    return [[25, 25, 25], [35, 35, 35], [45, 45, 45]]


def _render_tabs(strict_no_duplicate_hand):
    tab_replay, tab_simulate = st.tabs(["Match Replay", "Simulation Lab"])

    with tab_replay:
        _render_section_badge("Turn Builder")
        _render_turn_builder_simple(strict_no_duplicate_hand)

        _render_section_badge("Broadcast Replay")
        _render_replay(
            events_raw=st.session_state.events_raw,
            strict_no_duplicate_hand=strict_no_duplicate_hand,
            prefix="main",
        )

    with tab_simulate:
        _render_section_badge("Generate Simulated Games")
        _render_simulator_controls(strict_no_duplicate_hand)


_render_top_controls()
_render_tabs(bool(st.session_state.strict_no_duplicate_hand))
