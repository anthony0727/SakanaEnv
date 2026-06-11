"""
Three competing policies for the algorithm-fish demo.

Each fish runs a different decision-making strategy:
  - random_policy:  PPO placeholder (untrained = random)
  - boids_policy:   DQN placeholder (rule-based, greedy)
  - lookahead_policy: AB-MCTS proxy (best-of-N candidates with 1-step rollout)

The lookahead policy is the key: it simulates AB-MCTS's core advantage
(inference-time compute scaling) without needing the full treequest
library. It generates N candidate actions, evaluates each by rolling
forward one step, and picks the best. This is the "wider search = better
outcome" thesis of the AB-MCTS paper, made visible in a fish.
"""

from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp

from .env import EnvConfig, EnvState, step
from .boids import boids_action


def random_policy(state: EnvState, cfg: EnvConfig, key: jax.Array,
                  fish_idx: int) -> jnp.ndarray:
    """PPO placeholder — untrained = random actions."""
    return jax.random.uniform(
        key, (2,), minval=-cfg.fish_max_accel, maxval=cfg.fish_max_accel
    )


def boids_policy(state: EnvState, cfg: EnvConfig, key: jax.Array,
                 fish_idx: int) -> jnp.ndarray:
    """DQN placeholder — hand-crafted boids rules, greedy."""
    all_actions = boids_action(state, cfg)
    return all_actions[fish_idx]


def lookahead_policy(state: EnvState, cfg: EnvConfig, key: jax.Array,
                     fish_idx: int, n_candidates: int = 16) -> jnp.ndarray:
    """AB-MCTS proxy — best-of-N with 1-step lookahead, vectorized via vmap."""
    base = boids_action(state, cfg)[fish_idx]

    # food-biased direction
    fish_pos = state.fish_pos[fish_idx]
    food_deltas = state.food_pos - fish_pos
    food_sq = jnp.sum(food_deltas * food_deltas, axis=-1)
    food_sq = jnp.where(state.food_active, food_sq, 1e9)
    nearest_food = jnp.argmin(food_sq)
    food_dir = food_deltas[nearest_food]
    food_norm = jnp.linalg.norm(food_dir) + 1e-8
    food_accel = (food_dir / food_norm) * cfg.fish_max_accel

    # generate all candidates at once
    keys = jax.random.split(key, n_candidates + 1)
    noise = jax.random.normal(keys[0], (n_candidates, 2)) * cfg.fish_max_accel * 0.8
    # first candidate = pure food, next 3 = food + small noise, rest = boids + noise
    candidates = jnp.where(
        jnp.arange(n_candidates)[:, None] == 0,
        food_accel[None, :],
        jnp.where(
            jnp.arange(n_candidates)[:, None] < 4,
            food_accel[None, :] + noise * 0.4,
            base[None, :] + noise,
        )
    )
    # clip all
    norms = jnp.linalg.norm(candidates, axis=-1, keepdims=True) + 1e-8
    candidates = candidates * jnp.minimum(norms, cfg.fish_max_accel) / norms

    # vectorized 1-step rollout: vmap over candidates
    base_actions = boids_action(state, cfg)

    def eval_one(candidate):
        actions = base_actions.at[fish_idx].set(candidate)
        _, rewards, _ = step(state, actions, keys[-1], cfg)
        return rewards[fish_idx]

    all_rewards = jax.vmap(eval_one)(candidates)  # (n_candidates,)
    best_idx = jnp.argmax(all_rewards)
    return candidates[best_idx]


def competitive_step(state: EnvState, cfg: EnvConfig, key: jax.Array):
    """Run one step with 3 different policies (one per fish).

    Fish 0 = PPO (random)
    Fish 1 = DQN (boids)
    Fish 2 = AB-MCTS (lookahead best-of-8)
    """
    k1, k2, k3, k_step = jax.random.split(key, 4)

    a0 = random_policy(state, cfg, k1, 0)
    a1 = boids_policy(state, cfg, k2, 1)
    a2 = lookahead_policy(state, cfg, k3, 2, n_candidates=8)

    actions = jnp.stack([a0, a1, a2], axis=0)

    # pad if n_fish > 3
    if cfg.n_fish > 3:
        extra = jnp.zeros((cfg.n_fish - 3, 2))
        actions = jnp.concatenate([actions, extra], axis=0)

    return step(state, actions, k_step, cfg)
