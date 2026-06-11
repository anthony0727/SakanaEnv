"""
Boids baseline rollout — renders a GIF showing emergent schooling + predator
evasion + food seeking with the Reynolds rules as policy.

This is the baseline the AB-MCTS-M swarm-imitation experiment will measure
against.

Run:
    uv run python examples/run_boids.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import jax

from sakana_env import boids, env, viz


def main():
    cfg = env.EnvConfig(
        n_fish=48,
        n_predators=2,
        n_food=20,
        world_size=80.0,
        max_steps=240,
    )

    @jax.jit
    def step_fn(state, key):
        action = boids.boids_action(state, cfg)
        return env.step(state, action, key, cfg)

    key = jax.random.PRNGKey(7)
    state = env.reset(key, cfg)

    print(f"SakanaEnv boids rollout: {cfg.n_fish} fish, {cfg.n_predators} predators, "
          f"{cfg.max_steps} steps")
    t0 = time.time()

    states = [state]
    for _ in range(cfg.max_steps):
        key, sk = jax.random.split(key)
        state, reward, done = step_fn(state, sk)
        states.append(state)

    jax.block_until_ready(state.fish_pos)
    dt = time.time() - t0
    print(f"  rollout: {dt:.2f}s  ({cfg.max_steps / dt:,.0f} steps/sec, python-loop)")
    print(f"  survivors: {int(state.fish_alive.sum())}/{cfg.n_fish}")

    out_path = Path(__file__).resolve().parents[1] / "assets" / "boids.gif"
    print(f"rendering → {out_path}")
    viz.render_gif(states, cfg, str(out_path), fps=24, figsize=(7, 7), dpi=100)
    print("done")

    if sys.platform == "darwin":
        subprocess.run(["open", str(out_path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(out_path)], check=False)


if __name__ == "__main__":
    main()
