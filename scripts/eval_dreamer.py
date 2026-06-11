"""
Competitive eval: PPO vs DreamerV3 vs AB-MCTS.

Loads trained PPO (SB3) and DreamerV3 (sheeprl) checkpoints,
runs against zero-shot AB-MCTS in normal/whirlpool/current.

Usage:
    uv run python scripts/eval_dreamer.py
    uv run python scripts/eval_dreamer.py --dreamer-ckpt logs/runs/dreamer_v3/.../ckpt_200000_0.ckpt
"""

from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax
import jax.numpy as jnp

from sakana_env import env as sakana_env
from sakana_env.env import reset, step, observe
from sakana_env.policies import lookahead_policy
from sakana_env import viz_web


def find_latest_dreamer_ckpt():
    pattern = "logs/runs/dreamer_v3/SakanaEnv-v0/*/version_0/checkpoint/*.ckpt"
    ckpts = sorted(glob.glob(pattern))
    return ckpts[-1] if ckpts else None


def load_dreamer_agent(ckpt_path):
    """Load trained DreamerV3 from sheeprl checkpoint."""
    import torch
    from sheeprl.algos.dreamer_v3.agent import build_agent
    from sheeprl.utils.utils import dotdict
    from omegaconf import OmegaConf

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dotdict(OmegaConf.to_container(OmegaConf.create(ckpt.get("cfg", {})), resolve=True))

    # Build agent from config
    obs_space = {"obs": (27,)}  # vector obs
    agent, _ = build_agent(
        fabric=None,
        actions_dim=[2],
        is_continuous=True,
        cfg=cfg,
        obs_space=obs_space,
        world_model_state=ckpt.get("world_model"),
        actor_state=ckpt.get("actor"),
        critic_state=ckpt.get("critic"),
        target_critic_state=ckpt.get("target_critic"),
    )
    agent.eval()
    return agent, cfg


def dreamer_predict(agent, obs_np):
    """Get action from DreamerV3 agent given numpy obs."""
    import torch
    with torch.no_grad():
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        action = agent.get_greedy_action({"obs": obs_t})
    return np.array(action.squeeze(0).cpu())


def run_eval(cfg, ppo_model, dreamer_agent, action_table,
             n_episodes, episode_len, label, export_path=None):
    all_scores = np.zeros((n_episodes, 3))
    best_states = None

    for ep in range(n_episodes):
        key = jax.random.PRNGKey(ep)
        state = reset(key, cfg)
        states = [state] if ep == 0 and export_path else None
        scores = np.zeros(3)

        for t in range(episode_len):
            all_obs = observe(state, cfg)

            # PPO (fish 0)
            ppo_action, _ = ppo_model.predict(
                np.array(all_obs[0], dtype=np.float32), deterministic=True)

            # DreamerV3 (fish 1)
            dreamer_action = dreamer_predict(
                dreamer_agent, np.array(all_obs[1], dtype=np.float32))
            # scale from [-1,1] to accel range
            dreamer_action = dreamer_action * cfg.fish_max_accel

            # AB-MCTS (fish 2)
            key, k = jax.random.split(key)
            mcts_action = lookahead_policy(state, cfg, k, 2, n_candidates=16)

            actions = jnp.stack([
                jnp.array(ppo_action),
                jnp.array(dreamer_action),
                mcts_action,
            ])

            key, k = jax.random.split(key)
            state, rewards, done = step(state, actions, k, cfg)
            scores += np.array(rewards[:3])

            if states is not None:
                states.append(state)

        all_scores[ep] = scores
        if states is not None:
            best_states = states
        print(f"  ep {ep+1}/{n_episodes}: PPO={scores[0]:.1f} DV3={scores[1]:.1f} MCTS={scores[2]:.1f}")

    labels = ['PPO (trained)', 'DreamerV3 (trained)', 'AB-MCTS (zero-shot)']
    print(f"\n{'='*55}")
    print(f"{label}: {n_episodes} episodes x {episode_len} steps")
    print(f"{'='*55}")
    for i, lb in enumerate(labels):
        print(f"  {lb:30s}: {all_scores[:, i].mean():6.1f} +/- {all_scores[:, i].std():.1f}")
    print(f"{'='*55}")

    if best_states and export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        viz_web.export_trajectory(best_states, cfg, export_path)
        print(f"Exported -> {export_path}")

    return all_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamer-ckpt", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    # Load PPO
    ppo_model = PPO.load("assets/weights/ppo_fish0")

    # Load DreamerV3
    ckpt = args.dreamer_ckpt or find_latest_dreamer_ckpt()
    if not ckpt:
        print("ERROR: No DreamerV3 checkpoint found. Train first:")
        print("  PYTHONPATH=. uv run sheeprl --config-dir=configs exp=dreamer_v3_sakana")
        return
    print(f"Loading DreamerV3 from: {ckpt}")
    dreamer_agent, dreamer_cfg = load_dreamer_agent(ckpt)

    # DQN action table not needed — replaced by DreamerV3
    common = dict(
        n_fish=3, n_predators=0, n_food=32, world_size=30.0,
        fish_max_speed=2.0, fish_max_accel=4.0, sense_radius=50.0,
        k_neighbors=2, food_eat_radius=2.5, food_respawn_prob=0.15,
    )
    L = args.steps
    N = args.episodes

    cfg_n = sakana_env.EnvConfig(max_steps=L, **common)
    cfg_w = sakana_env.EnvConfig(max_steps=L, **common,
        whirlpool_active=True, whirlpool_x=15.0, whirlpool_y=15.0,
        whirlpool_strength=5.0, whirlpool_pull=2.5, whirlpool_radius=14.0)
    cfg_c = sakana_env.EnvConfig(max_steps=L, **common,
        current_active=True, current_dx=3.0, current_dy=1.0)

    run_eval(cfg_n, ppo_model, dreamer_agent, None, N, L, "NORMAL", "web/rollouts/boids.json")
    run_eval(cfg_w, ppo_model, dreamer_agent, None, N, L, "WHIRLPOOL", "web/rollouts/whirlpool.json")
    run_eval(cfg_c, ppo_model, dreamer_agent, None, N, L, "CURRENT", "web/rollouts/current.json")


if __name__ == "__main__":
    main()
