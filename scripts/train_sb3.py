"""
Train PPO and DQN agents using Stable-Baselines3.

This is the real training script — uses official SB3 implementations,
not hand-rolled. Run on Colab T4 via VSCode or locally.

Usage:
    # install deps first
    uv sync --extra sb3

    # train PPO (fish 0)
    uv run python scripts/train_sb3.py --algo ppo --fish 0 --steps 100000

    # train DQN (fish 1, discretized actions)
    uv run python scripts/train_sb3.py --algo dqn --fish 1 --steps 100000

    # competitive evaluation: PPO vs DQN vs AB-MCTS
    uv run python scripts/train_sb3.py --eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def train_ppo(fish_idx: int, total_timesteps: int):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from sakana_env.gym_wrapper import SakanaGymEnv

    # 4 parallel envs for faster data collection
    def make_env(seed):
        def _init():
            return SakanaGymEnv(fish_idx=fish_idx, seed=seed)
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(4)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    log_dir = str(Path(__file__).resolve().parents[1] / "logs")
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        tensorboard_log=log_dir,
    )
    print(f"Training PPO on fish[{fish_idx}] for {total_timesteps:,} steps (4 parallel envs)...")
    model.learn(total_timesteps=total_timesteps)

    out = WEIGHTS_DIR / f"ppo_fish{fish_idx}"
    model.save(str(out))
    vec_env.save(str(WEIGHTS_DIR / f"ppo_fish{fish_idx}_vecnorm.pkl"))
    print(f"Saved: {out}")
    return model


def train_dqn(fish_idx: int, total_timesteps: int):
    """DQN needs discrete actions. We wrap the continuous env."""
    from stable_baselines3 import DQN
    from sakana_env.gym_wrapper import SakanaGymEnv
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces

    # Discretize: 8 directions × 2 magnitudes = 16 actions
    import jax.numpy as jnp
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    mags = np.array([2.0, 4.0])
    ACTION_TABLE = np.stack([
        np.stack([np.cos(angles) * m, np.sin(angles) * m], axis=-1)
        for m in mags
    ]).reshape(-1, 2).astype(np.float32)

    class DiscreteSakanaEnv(gym.Wrapper):
        """Wraps continuous SakanaGymEnv with 16 discrete actions."""
        def __init__(self, fish_idx):
            super().__init__(SakanaGymEnv(fish_idx=fish_idx))
            self.action_space = spaces.Discrete(len(ACTION_TABLE))

        def step(self, action):
            continuous = ACTION_TABLE[action]
            return self.env.step(continuous)

    env = DiscreteSakanaEnv(fish_idx=fish_idx)
    log_dir = str(Path(__file__).resolve().parents[1] / "logs")
    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        train_freq=4,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        tensorboard_log=log_dir,
    )
    print(f"Training DQN on fish[{fish_idx}] for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps)

    out = WEIGHTS_DIR / f"dqn_fish{fish_idx}"
    model.save(str(out))
    print(f"Saved: {out}")
    return model


def _run_eval(cfg, ppo_model, dqn_model, action_table, n_episodes, episode_len, label, export_path=None):
    """Run competitive eval with given config. Returns (n_episodes, 3) scores."""
    from sakana_env.env import reset, step, observe
    from sakana_env.policies import lookahead_policy
    from sakana_env import viz_web
    import jax
    import jax.numpy as jnp
    import numpy as np

    all_scores = np.zeros((n_episodes, 3))
    best_states = None

    for ep in range(n_episodes):
        key = jax.random.PRNGKey(ep)
        state = reset(key, cfg)
        states = [state] if ep == 0 and export_path else None
        scores = np.zeros(3)

        for t in range(episode_len):
            all_obs = observe(state, cfg)
            ppo_action, _ = ppo_model.predict(np.array(all_obs[0], dtype=np.float32), deterministic=True)
            dqn_discrete, _ = dqn_model.predict(np.array(all_obs[1], dtype=np.float32), deterministic=True)
            dqn_action = action_table[int(dqn_discrete)]
            key, k = jax.random.split(key)
            mcts_action = lookahead_policy(state, cfg, k, 2, n_candidates=8)

            actions = jnp.stack([jnp.array(ppo_action), jnp.array(dqn_action), mcts_action])
            key, k = jax.random.split(key)
            state, rewards, done = step(state, actions, k, cfg)
            scores += np.array(rewards[:3])
            if states is not None:
                states.append(state)

        all_scores[ep] = scores
        if states is not None:
            best_states = states

    labels = ['PPO (trained)', 'DQN (trained)', 'AB-MCTS (zero-shot)']
    print(f"\n{'='*55}")
    print(f"{label}: {n_episodes} episodes × {episode_len} steps")
    print(f"{'='*55}")
    for i, lb in enumerate(labels):
        print(f"  {lb:30s}: {all_scores[:, i].mean():6.1f} ± {all_scores[:, i].std():.1f}")
    print(f"{'='*55}")

    if best_states and export_path:
        viz_web.export_trajectory(best_states, cfg, export_path)
        print(f"Exported → {export_path}")

    return all_scores


def competitive_eval(n_episodes: int = 50, episode_len: int = 500):
    """Deploy trained PPO, DQN, and AB-MCTS in normal + whirlpool conditions."""
    from stable_baselines3 import PPO, DQN
    from sakana_env import env
    import numpy as np

    common = dict(
        n_fish=3, n_predators=0, n_food=32,
        world_size=30.0, fish_max_speed=2.0, fish_max_accel=4.0,
        sense_radius=50.0, k_neighbors=2,
        food_eat_radius=2.5, food_respawn_prob=0.15,
    )
    long_len = episode_len * 4
    cfg_normal = env.EnvConfig(max_steps=long_len, **common)
    cfg_whirlpool = env.EnvConfig(
        max_steps=long_len, **common,
        whirlpool_active=True, whirlpool_x=15.0, whirlpool_y=15.0,
        whirlpool_strength=5.0, whirlpool_pull=2.5, whirlpool_radius=14.0,
    )
    cfg_current = env.EnvConfig(
        max_steps=long_len, **common,
        current_active=True, current_dx=3.0, current_dy=1.0,
    )

    ppo_path = WEIGHTS_DIR / "ppo_fish0"
    dqn_path = WEIGHTS_DIR / "dqn_fish1"
    if not ppo_path.with_suffix(".zip").exists() or not dqn_path.with_suffix(".zip").exists():
        print("ERROR: trained weights not found. Run training first.")
        return

    ppo_model = PPO.load(str(ppo_path))
    dqn_model = DQN.load(str(dqn_path))

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    mags = np.array([2.0, 4.0])
    action_table = np.stack([
        np.stack([np.cos(angles) * m, np.sin(angles) * m], axis=-1)
        for m in mags
    ]).reshape(-1, 2).astype(np.float32)

    _run_eval(cfg_normal, ppo_model, dqn_model, action_table,
              n_episodes, long_len, "NORMAL", "web/rollouts/boids.json")
    _run_eval(cfg_whirlpool, ppo_model, dqn_model, action_table,
              n_episodes, long_len, "WHIRLPOOL", "web/rollouts/whirlpool.json")
    _run_eval(cfg_current, ppo_model, dqn_model, action_table,
              n_episodes, long_len, "CURRENT", "web/rollouts/current.json")


def main():
    parser = argparse.ArgumentParser(description="Train/eval SakanaEnv agents with SB3")
    parser.add_argument("--algo", choices=["ppo", "dqn"], help="Algorithm to train")
    parser.add_argument("--fish", type=int, default=0, help="Fish index to control")
    parser.add_argument("--steps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--eval", action="store_true", help="Run competitive evaluation")
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()

    if args.eval:
        competitive_eval(n_episodes=args.eval_episodes)
    elif args.algo == "ppo":
        train_ppo(args.fish, args.steps)
    elif args.algo == "dqn":
        train_dqn(args.fish, args.steps)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
