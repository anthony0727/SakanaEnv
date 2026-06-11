"""
Minimal PPO and DQN training for SakanaEnv.

PPO: continuous actions, actor-critic MLP, GAE, clip surrogate.
DQN: discretized actions (16 directions), replay buffer, target network.
Both train a single fish (index 0) while others run boids.

Usage:
    from sakana_env.train import train_ppo, train_dqn
    ppo_params = train_ppo(cfg, n_episodes=500)
    dqn_params = train_dqn(cfg, n_episodes=500)
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .env import EnvConfig, EnvState, reset, step, observe, obs_dim
from .boids import boids_action


# =========================================================================
# Shared MLP utilities
# =========================================================================

def init_mlp(key, layer_sizes):
    """Initialize MLP params as list of (W, b) tuples."""
    params = []
    for i in range(len(layer_sizes) - 1):
        k1, k2, key = jax.random.split(key, 3)
        scale = jnp.sqrt(2.0 / layer_sizes[i])
        W = jax.random.normal(k1, (layer_sizes[i], layer_sizes[i+1])) * scale
        b = jnp.zeros(layer_sizes[i+1])
        params.append((W, b))
    return params


def forward_mlp(params, x):
    """Forward pass through MLP with ReLU hidden layers."""
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


# =========================================================================
# PPO — continuous actions
# =========================================================================

def _ppo_policy(actor_params, obs, key):
    """Sample continuous 2D action from Gaussian policy."""
    mean = jnp.tanh(forward_mlp(actor_params, obs)) * 4.0  # scale to accel range
    std = 0.5
    action = mean + jax.random.normal(key, mean.shape) * std
    log_prob = -0.5 * jnp.sum(((action - mean) / std) ** 2) - jnp.log(std) * 2
    return action, log_prob


def _ppo_value(critic_params, obs):
    return forward_mlp(critic_params, obs).squeeze(-1)


def train_ppo(
    cfg: EnvConfig,
    n_episodes: int = 300,
    episode_len: int = 100,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    fish_idx: int = 0,
    hidden: int = 64,
    seed: int = 0,
    verbose: bool = True,
):
    """Train PPO on a single fish. Returns (actor_params, critic_params)."""
    key = jax.random.PRNGKey(seed)
    od = obs_dim(cfg)

    # init networks
    k1, k2, key = jax.random.split(key, 3)
    actor_params = init_mlp(k1, [od, hidden, hidden, 2])
    critic_params = init_mlp(k2, [od, hidden, hidden, 1])

    # simple SGD (no optax dependency)
    def sgd_update(params, grads, lr):
        return [(W - lr * gW, b - lr * gb) for (W, b), (gW, gb) in zip(params, grads)]

    for ep in range(n_episodes):
        k_reset, key = jax.random.split(key)
        state = reset(k_reset, cfg)

        # collect trajectory
        obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []

        for t in range(episode_len):
            all_obs = observe(state, cfg)
            obs_i = all_obs[fish_idx]

            k_act, k_step, key = jax.random.split(key, 3)
            action_i, logp = _ppo_policy(actor_params, obs_i, k_act)
            value_i = _ppo_value(critic_params, obs_i)

            # other fish use boids
            all_actions = boids_action(state, cfg)
            all_actions = all_actions.at[fish_idx].set(action_i)

            state, rewards, done = step(state, all_actions, k_step, cfg)

            obs_buf.append(obs_i)
            act_buf.append(action_i)
            logp_buf.append(logp)
            rew_buf.append(rewards[fish_idx])
            val_buf.append(value_i)

        # convert to arrays
        obs_arr = jnp.stack(obs_buf)
        act_arr = jnp.stack(act_buf)
        logp_arr = jnp.stack(logp_buf)
        rew_arr = jnp.stack(rew_buf)
        val_arr = jnp.stack(val_buf)

        # GAE
        advantages = jnp.zeros(episode_len)
        gae = 0.0
        for t in reversed(range(episode_len)):
            next_val = val_arr[t + 1] if t + 1 < episode_len else 0.0
            delta = rew_arr[t] + gamma * next_val - val_arr[t]
            gae = delta + gamma * lam * gae
            advantages = advantages.at[t].set(gae)
        returns = advantages + val_arr

        # PPO update
        for _ in range(n_epochs):
            def ppo_loss(actor_p, critic_p):
                means = jnp.tanh(jax.vmap(forward_mlp, in_axes=(None, 0))(actor_p, obs_arr)) * 4.0
                std = 0.5
                new_logp = -0.5 * jnp.sum(((act_arr - means) / std) ** 2, axis=-1) - jnp.log(std) * 2
                ratio = jnp.exp(new_logp - logp_arr)
                clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))

                values = jax.vmap(forward_mlp, in_axes=(None, 0))(critic_p, obs_arr).squeeze(-1)
                critic_loss = jnp.mean((values - returns) ** 2)
                return actor_loss + 0.5 * critic_loss

            loss, (a_grads, c_grads) = jax.value_and_grad(ppo_loss, argnums=(0, 1))(actor_params, critic_params)
            actor_params = sgd_update(actor_params, a_grads, lr)
            critic_params = sgd_update(critic_params, c_grads, lr)

        if verbose and (ep + 1) % 50 == 0:
            total_reward = float(rew_arr.sum())
            print(f"  PPO ep {ep+1}/{n_episodes}: reward={total_reward:.1f}, loss={float(loss):.4f}")

    return actor_params, critic_params


# =========================================================================
# DQN — discretized actions
# =========================================================================

# 8 directions × 2 magnitudes = 16 discrete actions
_N_ACTIONS = 16
_ANGLES = jnp.linspace(0, 2 * jnp.pi, 8, endpoint=False)
_MAGS = jnp.array([2.0, 4.0])
_ACTION_TABLE = jnp.stack([
    jnp.concatenate([jnp.cos(_ANGLES) * m, jnp.sin(_ANGLES) * m]).reshape(2, -1).T
    for m in _MAGS
]).reshape(-1, 2)  # (16, 2)


def train_dqn(
    cfg: EnvConfig,
    n_episodes: int = 300,
    episode_len: int = 100,
    lr: float = 1e-3,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    buffer_size: int = 5000,
    batch_size: int = 64,
    target_update: int = 20,
    fish_idx: int = 0,
    hidden: int = 64,
    seed: int = 1,
    verbose: bool = True,
):
    """Train DQN on a single fish with discretized actions. Returns q_params."""
    key = jax.random.PRNGKey(seed)
    od = obs_dim(cfg)

    k1, key = jax.random.split(key)
    q_params = init_mlp(k1, [od, hidden, hidden, _N_ACTIONS])
    target_params = q_params  # copy

    # replay buffer (simple numpy ring buffer)
    buf_obs = np.zeros((buffer_size, od), dtype=np.float32)
    buf_act = np.zeros(buffer_size, dtype=np.int32)
    buf_rew = np.zeros(buffer_size, dtype=np.float32)
    buf_next = np.zeros((buffer_size, od), dtype=np.float32)
    buf_done = np.zeros(buffer_size, dtype=np.float32)
    buf_ptr = 0
    buf_len = 0

    def sgd_update(params, grads, lr):
        return [(W - lr * gW, b - lr * gb) for (W, b), (gW, gb) in zip(params, grads)]

    for ep in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / n_episodes)
        k_reset, key = jax.random.split(key)
        state = reset(k_reset, cfg)

        ep_reward = 0.0
        for t in range(episode_len):
            all_obs = observe(state, cfg)
            obs_i = all_obs[fish_idx]

            # epsilon-greedy
            k_eps, k_step, key = jax.random.split(key, 3)
            if jax.random.uniform(k_eps) < eps:
                act_idx = int(jax.random.randint(k_eps, (), 0, _N_ACTIONS))
            else:
                q_vals = forward_mlp(q_params, obs_i)
                act_idx = int(jnp.argmax(q_vals))

            action_i = _ACTION_TABLE[act_idx]

            all_actions = boids_action(state, cfg)
            all_actions = all_actions.at[fish_idx].set(action_i)
            state, rewards, done = step(state, all_actions, k_step, cfg)

            next_obs_i = observe(state, cfg)[fish_idx]
            r = float(rewards[fish_idx])
            ep_reward += r

            # store transition
            idx = buf_ptr % buffer_size
            buf_obs[idx] = np.array(obs_i)
            buf_act[idx] = act_idx
            buf_rew[idx] = r
            buf_next[idx] = np.array(next_obs_i)
            buf_done[idx] = 0.0
            buf_ptr += 1
            buf_len = min(buf_len + 1, buffer_size)

            # train on mini-batch
            if buf_len >= batch_size:
                idxs = np.random.choice(buf_len, batch_size, replace=False)
                b_obs = jnp.array(buf_obs[idxs])
                b_act = jnp.array(buf_act[idxs])
                b_rew = jnp.array(buf_rew[idxs])
                b_next = jnp.array(buf_next[idxs])

                def dqn_loss(params):
                    q = jax.vmap(forward_mlp, in_axes=(None, 0))(params, b_obs)
                    q_selected = q[jnp.arange(batch_size), b_act]
                    q_next = jax.vmap(forward_mlp, in_axes=(None, 0))(target_params, b_next)
                    target = b_rew + gamma * jnp.max(q_next, axis=-1)
                    return jnp.mean((q_selected - target) ** 2)

                loss, grads = jax.value_and_grad(dqn_loss)(q_params)
                q_params = sgd_update(q_params, grads, lr)

        # target network update
        if (ep + 1) % target_update == 0:
            target_params = q_params

        if verbose and (ep + 1) % 50 == 0:
            print(f"  DQN ep {ep+1}/{n_episodes}: reward={ep_reward:.1f}, eps={eps:.2f}")

    return q_params


# =========================================================================
# Deploy trained policies in competitive rollout
# =========================================================================

def trained_ppo_action(actor_params, obs):
    """Deterministic PPO action (no noise at eval time)."""
    return jnp.tanh(forward_mlp(actor_params, obs)) * 4.0


def trained_dqn_action(q_params, obs):
    """Greedy DQN action."""
    q_vals = forward_mlp(q_params, obs)
    return _ACTION_TABLE[jnp.argmax(q_vals)]
