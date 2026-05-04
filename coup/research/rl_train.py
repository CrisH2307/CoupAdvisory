from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path

from coup.research.heuristic_baselines import benchmark_heuristic_policies
from coup.research.rl_env import OfflineCoupRLEnv


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done):
        self.data.append((list(state), int(action), float(reward), list(next_state), bool(done)))

    def sample(self, batch_size, rng):
        batch_size = min(int(batch_size), len(self.data))
        return rng.sample(list(self.data), batch_size)

    def __len__(self):
        return len(self.data)


class DQNPolicy:
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self, in_dim, out_dim, hid_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hid_dim),
                    nn.ReLU(),
                    nn.Linear(hid_dim, hid_dim),
                    nn.ReLU(),
                    nn.Linear(hid_dim, out_dim),
                )

            def forward(self, x):
                return self.net(x)

        self.torch = torch
        self.model = _Model(int(state_dim), int(action_dim), int(hidden_dim))
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_labels = []

    def to(self, device):
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, payload):
        self.model.load_state_dict(payload)

    def eval(self):
        self.model.eval()
        return self

    def train(self):
        self.model.train()
        return self

    def q_values(self, state_batch):
        tensor = self.torch.tensor(state_batch, dtype=self.torch.float32)
        return self.model(tensor)

    def act(self, state):
        with self.torch.no_grad():
            tensor = self.torch.tensor(state, dtype=self.torch.float32).unsqueeze(0)
            q = self.model(tensor)
            return int(self.torch.argmax(q, dim=1).item())


def train_dqn_policy(
    transitions_csv,
    out_dir,
    *,
    episodes=200,
    seed=42,
    gamma=0.98,
    learning_rate=1e-3,
    batch_size=64,
    buffer_size=10000,
    warmup_steps=256,
    target_update_steps=200,
    hidden_dim=128,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    exploration_mode="epsilon_greedy",
    tau_start=1.0,
    tau_end=0.1,
    tau_decay=0.995,
    compare_bc_dir=None,
    reward_mode="shaped",
):
    if int(episodes) < 1:
        raise ValueError("episodes must be >= 1")
    if exploration_mode not in ("epsilon_greedy", "softmax"):
        raise ValueError(
            f"exploration_mode must be 'epsilon_greedy' or 'softmax', got {exploration_mode!r}"
        )

    import numpy as np
    import torch

    transitions_csv = Path(transitions_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = OfflineCoupRLEnv.from_csv(transitions_csv, reward_mode=reward_mode)
    state_dim = env.state_dim()
    action_dim = env.action_dim()
    if state_dim <= 0 or action_dim <= 0:
        raise ValueError("Transition dataset does not contain valid state/action dimensions")

    policy = DQNPolicy(state_dim, action_dim, hidden_dim=hidden_dim)
    target = DQNPolicy(state_dim, action_dim, hidden_dim=hidden_dim)
    target.load_state_dict(policy.state_dict())
    policy.action_labels = list(env.action_labels)
    target.action_labels = list(env.action_labels)

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
    loss_fn = torch.nn.MSELoss()

    replay = ReplayBuffer(buffer_size)
    rng = random.Random(seed)

    epsilon = float(epsilon_start)
    tau = float(tau_start)
    global_steps = 0
    history_rows = []
    action_counts = {label: 0 for label in env.action_labels}
    observed_states = []

    total_reward_sum = 0.0

    game_ids = env.available_games()
    for episode in range(1, int(episodes) + 1):
        game_id = game_ids[(episode - 1) % len(game_ids)] if game_ids else None
        state = env.reset(game_id=game_id)

        done = False
        episode_reward = 0.0
        episode_steps = 0
        episode_entropy_sum = 0.0

        while not done and episode_steps < 10_000:
            observed_states.append(list(state))

            if exploration_mode == "epsilon_greedy":
                if rng.random() < epsilon:
                    action_index = rng.randrange(action_dim)
                else:
                    action_index = policy.act(state)
                p_greedy = 1.0 - epsilon + epsilon / action_dim
                p_other = epsilon / action_dim
                step_entropy = float(
                    -(p_greedy * np.log(p_greedy + 1e-12)
                      + (action_dim - 1) * p_other * np.log(p_other + 1e-12))
                )
            else:
                with torch.no_grad():
                    q_arr = policy.model(
                        torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    ).squeeze(0).numpy()
                scaled = q_arr / max(float(tau), 1e-6)
                scaled = scaled - float(np.max(scaled))
                probs = np.exp(scaled)
                probs = probs / probs.sum()
                action_index = int(
                    rng.choices(range(action_dim), weights=probs.tolist(), k=1)[0]
                )
                step_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

            episode_entropy_sum += step_entropy
            action_label = env.action_labels[action_index]
            action_counts[action_label] = int(action_counts.get(action_label, 0)) + 1

            step = env.step(action_index)
            replay.push(state, action_index, step.reward, step.next_state, step.done)

            state = list(step.next_state)
            episode_reward += float(step.reward)
            episode_steps += 1
            global_steps += 1
            done = bool(step.done)

            if len(replay) >= int(batch_size) and len(replay) >= int(warmup_steps):
                batch = replay.sample(batch_size, rng)
                s_batch = torch.tensor([item[0] for item in batch], dtype=torch.float32)
                a_batch = torch.tensor([item[1] for item in batch], dtype=torch.long)
                r_batch = torch.tensor([item[2] for item in batch], dtype=torch.float32)
                ns_batch = torch.tensor([item[3] for item in batch], dtype=torch.float32)
                d_batch = torch.tensor([item[4] for item in batch], dtype=torch.float32)

                q_values = policy.model(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target.model(ns_batch).max(dim=1).values
                    target_q = r_batch + float(gamma) * (1.0 - d_batch) * max_next_q

                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if int(target_update_steps) > 0 and global_steps % int(target_update_steps) == 0:
                target.load_state_dict(policy.state_dict())

        total_reward_sum += episode_reward
        avg_reward_so_far = total_reward_sum / episode
        avg_entropy = episode_entropy_sum / max(1, episode_steps)
        history_rows.append(
            {
                "episode": int(episode),
                "reward": float(episode_reward),
                "steps": int(episode_steps),
                "epsilon": float(epsilon),
                "tau": float(tau),
                "exploration_mode": exploration_mode,
                "action_entropy": float(avg_entropy),
                "avg_reward_so_far": float(avg_reward_so_far),
            }
        )

        epsilon = max(float(epsilon_end), float(epsilon) * float(epsilon_decay))
        tau = max(float(tau_end), float(tau) * float(tau_decay))

    model_path = out_dir / "dqn_model.pt"
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "hidden_dim": int(hidden_dim),
            "action_labels": list(env.action_labels),
            "exploration_mode": exploration_mode,
        },
        model_path,
    )

    train_metrics = {
        "episodes": int(episodes),
        "avg_episode_reward": float(sum(row["reward"] for row in history_rows) / len(history_rows)),
        "avg_episode_steps": float(sum(row["steps"] for row in history_rows) / len(history_rows)),
        "final_epsilon": float(epsilon),
        "final_tau": float(tau),
        "exploration_mode": exploration_mode,
        "avg_action_entropy": float(
            sum(row["action_entropy"] for row in history_rows) / len(history_rows)
        ),
    }

    _write_csv(out_dir / "train_episode_history.csv", history_rows)
    (out_dir / "train_action_counts.json").write_text(json.dumps(action_counts, indent=2))

    state_summary = _summarize_states(observed_states, state_dim)
    (out_dir / "train_state_summary.json").write_text(json.dumps(state_summary, indent=2))

    benchmark = benchmark_dqn_vs_random(
        transitions_csv=transitions_csv,
        model_path=model_path,
        seed=seed,
        reward_mode=reward_mode,
    )

    if compare_bc_dir:
        if reward_mode == "win":
            try:
                bc_stats = _score_bc_under_env(
                    transitions_csv=transitions_csv,
                    bc_dir=Path(compare_bc_dir),
                    reward_mode=reward_mode,
                )
                if bc_stats is not None:
                    benchmark["behavior_cloning"] = bc_stats
            except Exception:
                pass
        else:
            compare_path = Path(compare_bc_dir) / "benchmark.json"
            if compare_path.exists():
                try:
                    bc_data = json.loads(compare_path.read_text())
                    if "behavior_cloning" in bc_data:
                        benchmark["behavior_cloning"] = bc_data["behavior_cloning"]
                except Exception:
                    pass

    rollout_rows = build_policy_rollout_trace(
        transitions_csv=transitions_csv,
        model_path=model_path,
        seed=seed,
    )
    _write_csv(out_dir / "dqn_policy_rollout_trace.csv", rollout_rows)

    (out_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2))
    (out_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2))

    summary = {
        "train_metrics": train_metrics,
        "benchmark": benchmark,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_dqn_policy(model_path):
    import torch

    payload = torch.load(Path(model_path), map_location="cpu")
    model = DQNPolicy(
        payload["state_dim"],
        payload["action_dim"],
        hidden_dim=payload.get("hidden_dim", 128),
    )
    model.load_state_dict(payload["state_dict"])
    model.action_labels = list(payload.get("action_labels", []))
    model.eval()
    return model


def build_policy_rollout_trace(*, transitions_csv, model_path, seed=42):
    import numpy as np

    del seed

    env = OfflineCoupRLEnv.from_csv(transitions_csv)
    model = load_dqn_policy(model_path)

    rows = []
    for game_id in env.available_games():
        state = env.reset(game_id=game_id)
        done = False
        while not done:
            action_index = model.act(state)
            chosen_action = env.action_labels[action_index]
            step = env.step(action_index)
            rows.append(
                {
                    "game_id": game_id,
                    "transition_index": int(step.info.get("transition_index", 0)),
                    "chosen_action": chosen_action,
                    "logged_action": step.info.get("logged_action"),
                    "action_match": int(step.info.get("action_match", 0)),
                    "reward": float(step.reward),
                    "state_mean": float(np.mean(state)) if state else 0.0,
                    "state_norm": float(np.linalg.norm(state)) if state else 0.0,
                }
            )
            state = list(step.next_state)
            done = bool(step.done)

    return rows


def benchmark_dqn_vs_random(*, transitions_csv, model_path, seed=42, reward_mode="shaped"):
    env_dqn = OfflineCoupRLEnv.from_csv(transitions_csv, reward_mode=reward_mode)
    env_random = OfflineCoupRLEnv.from_csv(transitions_csv, reward_mode=reward_mode)
    model = load_dqn_policy(model_path)

    dqn_stats = _rollout_stats(env_dqn, choose_action=lambda state, env: model.act(state))

    rng = random.Random(seed)
    random_stats = _rollout_stats(
        env_random,
        choose_action=lambda state, env: rng.randrange(env.action_dim()),
    )

    baseline_stats = benchmark_heuristic_policies(transitions_csv, reward_mode=reward_mode)
    honest_stats = _score_honest_under_env(
        transitions_csv=transitions_csv, reward_mode=reward_mode
    )
    result = {
        "dqn_policy": dqn_stats,
        "random_policy": random_stats,
        "honest_policy": honest_stats,
    }
    result.update(baseline_stats)
    return result


def _score_bc_under_env(*, transitions_csv, bc_dir, reward_mode):
    import pickle

    import pandas as pd

    model_path = Path(bc_dir) / "bc_model.pkl"
    if not model_path.exists():
        return None
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    env = OfflineCoupRLEnv.from_csv(transitions_csv, reward_mode=reward_mode)
    frame = pd.read_csv(transitions_csv)
    predictions = model.predict(frame[feature_cols])
    pred_by_game = {}
    for game_id, pred in zip(frame["meta_game_id"].astype(str), predictions):
        pred_by_game.setdefault(game_id, []).append(str(pred))
    cursor = {g: 0 for g in pred_by_game}
    action_label_to_idx = {label: idx for idx, label in enumerate(env.action_labels)}

    def chooser(state, env):
        gid = env.current_game_id
        i = cursor[gid]
        cursor[gid] = i + 1
        label = pred_by_game[gid][i]
        return action_label_to_idx.get(label, 0)

    return _rollout_stats(env, choose_action=chooser)


def _score_honest_under_env(*, transitions_csv, reward_mode):
    env = OfflineCoupRLEnv.from_csv(transitions_csv, reward_mode=reward_mode)
    income_idx = env.action_labels.index("Income") if "Income" in env.action_labels else 0
    tax_idx = env.action_labels.index("Tax") if "Tax" in env.action_labels else income_idx
    coup_idx = env.action_labels.index("Coup") if "Coup" in env.action_labels else income_idx
    assass_idx = env.action_labels.index("Assassinate") if "Assassinate" in env.action_labels else income_idx

    def chooser(state, env):
        actor_coins = 0.0
        if env.current_rows and 0 <= env.cursor < len(env.current_rows):
            actor_coins = float(env.current_rows[env.cursor].get("s_actor_coins", 0.0))
        if actor_coins >= 10.0:
            return coup_idx
        if actor_coins >= 7.0:
            return coup_idx
        if actor_coins >= 3.0:
            return assass_idx
        return tax_idx

    return _rollout_stats(env, choose_action=chooser)


def _rollout_stats(env, *, choose_action):
    total_reward = 0.0
    total_steps = 0
    total_matches = 0
    episodes = 0

    for game_id in env.available_games():
        episodes += 1
        state = env.reset(game_id=game_id)
        done = False
        while not done:
            action_index = choose_action(state, env)
            step = env.step(action_index)
            total_reward += float(step.reward)
            total_steps += 1
            total_matches += int(step.info.get("action_match", 0))
            state = list(step.next_state)
            done = bool(step.done)

    if episodes == 0:
        return {
            "avg_reward_per_episode": 0.0,
            "avg_reward_per_step": 0.0,
            "action_match_rate": 0.0,
        }

    return {
        "avg_reward_per_episode": float(total_reward / episodes),
        "avg_reward_per_step": float(total_reward / total_steps) if total_steps else 0.0,
        "action_match_rate": float(total_matches / total_steps) if total_steps else 0.0,
    }


def _summarize_states(states, state_dim):
    import numpy as np

    if not states:
        return {
            "state_dim": int(state_dim),
            "observations": 0,
            "state_mean": [0.0 for _ in range(int(state_dim))],
            "state_std": [0.0 for _ in range(int(state_dim))],
        }

    array = np.asarray(states, dtype=float)
    return {
        "state_dim": int(array.shape[1]),
        "observations": int(array.shape[0]),
        "state_mean": [float(value) for value in np.mean(array, axis=0)],
        "state_std": [float(value) for value in np.std(array, axis=0)],
    }


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Train DQN policy from offline Markov transitions")
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--target-update-steps", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument(
        "--exploration-mode",
        choices=["epsilon_greedy", "softmax"],
        default="epsilon_greedy",
        help="Action-selection rule during training. 'softmax' uses an "
             "entropy-regularised Boltzmann policy over Q-values with annealed temperature.",
    )
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.1)
    parser.add_argument("--tau-decay", type=float, default=0.995)
    parser.add_argument("--compare-bc", default=None)
    parser.add_argument(
        "--reward-mode",
        choices=["shaped", "win"],
        default="shaped",
        help="shaped keeps legacy Δcoins+Δalive+match; win uses sparse terminal win reward.",
    )
    args = parser.parse_args()

    summary = train_dqn_policy(
        transitions_csv=args.transitions,
        out_dir=args.out,
        episodes=args.episodes,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        warmup_steps=args.warmup_steps,
        target_update_steps=args.target_update_steps,
        hidden_dim=args.hidden_dim,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        exploration_mode=args.exploration_mode,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_decay=args.tau_decay,
        compare_bc_dir=args.compare_bc,
        reward_mode=args.reward_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
