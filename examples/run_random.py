"""
Smoke test — random action policy.

The point of this script is not to demonstrate good fish behavior. It is to
verify end-to-end that:
    1. The JAX env resets + steps without error
    2. The observation shapes are internally consistent
    3. The matplotlib renderer produces a GIF
    4. The sakana-style fish silhouettes render visibly on the ink-on-rice-paper bg

Expected outcome: fish jitter around, predators chase the nearest one, most
fish die by the end of the episode. The resulting GIF shows the dev-mode
visual language (monochrome vector, no learning involved).

Run:
    uv sync
    uv run python examples/run_random.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from sakana_env import env, viz


def main():
    cfg = env.EnvConfig(
        n_fish=48,
        n_predators=2,
        n_food=16,
        world_size=80.0,
        max_steps=200,
    )

    key = jax.random.PRNGKey(0)
    state = env.reset(key, cfg)

    @jax.jit
    def step_fn(state, key):
        key, sk_action, sk_step = jax.random.split(key, 3)
        # random accelerations in [-fish_max_accel, +fish_max_accel]
        action = jax.random.uniform(
            sk_action,
            (cfg.n_fish, 2),
            minval=-cfg.fish_max_accel,
            maxval=cfg.fish_max_accel,
        )
        state, reward, done = env.step(state, action, sk_step, cfg)
        return state, reward, done, key

    print(f"SakanaEnv random rollout: {cfg.n_fish} fish, {cfg.n_predators} predators, "
          f"{cfg.max_steps} steps")
    t0 = time.time()

    states = [state]
    for t in range(cfg.max_steps):
        state, reward, done, key = step_fn(state, key)
        states.append(state)

    jax.block_until_ready(state.fish_pos)
    dt = time.time() - t0
    steps_per_sec = cfg.max_steps / dt
    n_alive = int(state.fish_alive.sum())
    print(f"  rollout time: {dt:.2f}s  ({steps_per_sec:,.0f} steps/sec)")
    print(f"  survivors:    {n_alive}/{cfg.n_fish}")

    out_path = Path(__file__).resolve().parents[1] / "assets" / "random.gif"
    print(f"rendering → {out_path}")
    viz.render_gif(states, cfg, str(out_path), fps=24, figsize=(7, 7), dpi=90)
    print("done")

    # Open the GIF in the default viewer
    if sys.platform == "darwin":
        subprocess.run(["open", str(out_path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(out_path)], check=False)


if __name__ == "__main__":
    main()
