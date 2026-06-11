"""
Honest steps/sec benchmark — scan-fused rollout.

Prints:
    - rollout time
    - steps/sec (single env)
    - steps/sec (batched across 32 parallel envs via vmap)
"""

from __future__ import annotations

import time

import jax

from sakana_env import env, rollout


def main():
    cfg = env.EnvConfig(
        n_fish=64,
        n_predators=2,
        n_food=32,
        world_size=100.0,
        max_steps=500,
    )

    scan_fn = rollout.make_scan_rollout(cfg, rollout.random_action)

    # -- warmup (first call is compile cost) --
    key = jax.random.PRNGKey(0)
    final, rs, ds = scan_fn(key, 500)
    jax.block_until_ready(final.fish_pos)
    print("warm-up done (compile cost excluded from timings)")

    # -- single env, 10K steps --
    t0 = time.time()
    n_steps = 10_000
    final, rs, ds = scan_fn(jax.random.PRNGKey(1), n_steps)
    jax.block_until_ready(final.fish_pos)
    dt = time.time() - t0
    print(f"\nsingle env, {n_steps:,} steps")
    print(f"  time: {dt:.3f}s")
    print(f"  rate: {n_steps / dt:,.0f} steps/sec")

    # -- batched via vmap --
    B = 32
    batched = jax.jit(jax.vmap(scan_fn, in_axes=(0, None)), static_argnums=(1,))
    keys = jax.random.split(jax.random.PRNGKey(2), B)
    # warmup
    warm_state, _, _ = batched(keys, 100)
    jax.block_until_ready(warm_state)

    t0 = time.time()
    final_b, rs_b, ds_b = batched(keys, 1_000)
    jax.block_until_ready(final_b)
    dt = time.time() - t0
    total = B * 1_000
    print(f"\nbatched {B} envs x 1,000 steps = {total:,} total")
    print(f"  time: {dt:.3f}s")
    print(f"  rate: {total / dt:,.0f} steps/sec")


if __name__ == "__main__":
    main()
