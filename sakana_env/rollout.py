"""
Scan-fused rollout — honest steps/sec benchmarks for JAX.

Using a Python for-loop around jax.jit(step) pays compile+dispatch overhead
per step. Wrapping the rollout in jax.lax.scan fuses the entire loop into
a single XLA graph, which is what gets you 100K+ steps/sec on CPU and
1M+ on GPU.

Use this for training loops + benchmarking. Use the Python loop for
examples where you want per-step access (rendering, debug logging).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .env import EnvConfig, EnvState, reset, step


ActionFn = Callable[[EnvState, EnvConfig, jax.Array], jnp.ndarray]


def make_scan_rollout(cfg: EnvConfig, action_fn: ActionFn):
    """Return a jit-compiled function that runs `n_steps` in a single scan.

    The returned fn:
        scan_fn(key, n_steps) -> (final_state, stacked_rewards, stacked_dones)

    Where `stacked_rewards` has shape (n_steps, n_fish) and `stacked_dones`
    has shape (n_steps,).
    """

    def _one_step(state, step_key):
        act_key, step_key_inner = jax.random.split(step_key)
        action = action_fn(state, cfg, act_key)
        next_state, reward, done = step(state, action, step_key_inner, cfg)
        return next_state, (reward, done)

    def scan_fn(key, n_steps: int):
        reset_key, scan_key = jax.random.split(key)
        init_state = reset(reset_key, cfg)
        step_keys = jax.random.split(scan_key, n_steps)
        final_state, (rewards, dones) = jax.lax.scan(_one_step, init_state, step_keys)
        return final_state, rewards, dones

    return jax.jit(scan_fn, static_argnums=(1,))


def random_action(state: EnvState, cfg: EnvConfig, key: jax.Array) -> jnp.ndarray:
    """Reference action function for benchmarks."""
    return jax.random.uniform(
        key,
        (cfg.n_fish, 2),
        minval=-cfg.fish_max_accel,
        maxval=cfg.fish_max_accel,
    )


__all__ = ["make_scan_rollout", "random_action"]
