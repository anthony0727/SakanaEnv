"""
SakanaEnv: decentralized swarm under partial observability.

A JAX environment where N fish navigate a 2D ocean with K predators and
M food patches. Each fish senses a local neighborhood only — no broadcast
channel, no hard-coded hierarchy. The coordination mechanism is whatever
the agent's decision layer provides.

The research question this env poses:
    Does treequest's AB-MCTS-M produce emergent context-sensitive
    leadership in a decentralized swarm, or does the Bayesian budget
    allocation flatten into uniform consensus?

The env itself is coordination-mechanism agnostic. Plug in boids, PPO,
AB-MCTS-M, or anything that maps (obs -> action). The env measures the
outcome; you decide what mechanism to test.

Usage:
    from sakana_env import env
    cfg = env.EnvConfig()
    key = jax.random.PRNGKey(0)
    state = env.reset(key, cfg)
    for t in range(cfg.max_steps):
        action = policy(env.observe(state, cfg))        # (N_fish, 2)
        key, sk = jax.random.split(key)
        state, reward, done = env.step(state, action, sk, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class EnvConfig:
    """Static environment configuration. Not part of the state PyTree."""

    # population
    n_fish: int = 64
    n_predators: int = 2
    n_food: int = 32

    # world
    world_size: float = 100.0
    dt: float = 0.1
    max_steps: int = 500

    # reward history window (for Level 2 swarm imitation observations)
    reward_window: int = 8

    # body radii — used by the position-based dynamics collision resolver
    # Bodies are circles; collisions push centers apart non-destructively.
    # Tails and fins can still overlap visually (they're outside this radius).
    fish_body_radius: float = 0.9
    pred_body_radius: float = 1.6
    collision_iters: int = 3  # PBD relaxation iterations per step

    # fish dynamics
    fish_max_speed: float = 2.0
    fish_max_accel: float = 4.0
    fish_drag: float = 0.05

    # predator dynamics
    pred_max_speed: float = 2.5
    pred_drag: float = 0.05
    pred_kill_radius: float = 1.0
    pred_chase_gain: float = 1.5  # how aggressively predators steer toward nearest fish

    # food
    food_eat_radius: float = 1.5
    food_respawn_prob: float = 0.01  # per timestep per dead food patch

    # sensing
    sense_radius: float = 15.0
    k_neighbors: int = 7  # must be < n_fish

    # reward shaping
    reward_food: float = 1.0
    reward_death: float = -10.0
    reward_schooling: float = 0.0   # DISABLED for competitive play (incentivizes clustering)
    reward_time_penalty: float = -0.002  # small urgency to eat faster

    # observation: how many nearest food items to include
    n_food_obs: int = 5

    # whirlpool — test-time perturbation (off during training)
    whirlpool_active: bool = False
    whirlpool_x: float = 15.0
    whirlpool_y: float = 15.0
    whirlpool_strength: float = 3.0   # max tangential accel at center
    whirlpool_pull: float = 1.5       # radial inward pull
    whirlpool_radius: float = 12.0    # effect radius

    # current — global directional force (off during training)
    current_active: bool = False
    current_dx: float = 3.0           # force direction x
    current_dy: float = 1.0           # force direction y


# =============================================================================
# State (PyTree)
# =============================================================================


class EnvState(NamedTuple):
    """Full environment state. All fields are jax arrays (leaves in the PyTree)."""

    # fish (N_fish,)
    fish_pos: jnp.ndarray  # (N_fish, 2)
    fish_vel: jnp.ndarray  # (N_fish, 2)
    fish_alive: jnp.ndarray  # (N_fish,) bool

    # per-fish sliding-window reward history (for Level 2 neighbor imitation)
    # most recent reward in index 0, oldest in index -1
    fish_recent_rewards: jnp.ndarray  # (N_fish, reward_window)

    # predators (N_pred,)
    pred_pos: jnp.ndarray  # (N_pred, 2)
    pred_vel: jnp.ndarray  # (N_pred, 2)

    # food (N_food,)
    food_pos: jnp.ndarray  # (N_food, 2)
    food_active: jnp.ndarray  # (N_food,) bool

    # bookkeeping
    t: jnp.ndarray  # () int32
    key: jnp.ndarray  # rng state carried in the pytree for pure-functional stepping


# =============================================================================
# Reset
# =============================================================================


def reset(key: jax.Array, cfg: EnvConfig) -> EnvState:
    """Initialize a fresh episode. All randomness flows from `key`."""
    k_fish, k_pred, k_food, k_state = jax.random.split(key, 4)

    fish_pos = jax.random.uniform(
        k_fish, (cfg.n_fish, 2), minval=0.0, maxval=cfg.world_size
    )
    pred_pos = jax.random.uniform(
        k_pred, (cfg.n_predators, 2), minval=0.0, maxval=cfg.world_size
    )
    food_pos = jax.random.uniform(
        k_food, (cfg.n_food, 2), minval=0.0, maxval=cfg.world_size
    )

    return EnvState(
        fish_pos=fish_pos,
        fish_vel=jnp.zeros((cfg.n_fish, 2)),
        fish_alive=jnp.ones((cfg.n_fish,), dtype=jnp.bool_),
        fish_recent_rewards=jnp.zeros((cfg.n_fish, cfg.reward_window)),
        pred_pos=pred_pos,
        pred_vel=jnp.zeros((cfg.n_predators, 2)),
        food_pos=food_pos,
        food_active=jnp.ones((cfg.n_food,), dtype=jnp.bool_),
        t=jnp.int32(0),
        key=k_state,
    )


# =============================================================================
# Observation
# =============================================================================


# per-fish observation layout:
#   [own_pos_normalized (2)]           ← NEW: spatial awareness
#   [own_vel (2)]
#   [k_neighbors * (rel_pos + rel_vel) = 4*k]
#   [n_food_obs * (rel_pos + visible) = 3*n_food_obs]  ← EXPANDED
#   (predator dims removed for n_predators=0)
# total = 4 + 4*k + 3*n_food_obs + (4 if n_predators>0 else 0)


def obs_dim(cfg: EnvConfig) -> int:
    pred_dim = 4 if cfg.n_predators > 0 else 0
    return 4 + 4 * cfg.k_neighbors + 3 * cfg.n_food_obs + pred_dim


def observe(state: EnvState, cfg: EnvConfig) -> jnp.ndarray:
    """Compute (N_fish, obs_dim) local observations.

    Each fish sees:
      - its own velocity
      - K nearest alive neighbors' relative positions + velocities
      - nearest food's relative position + a mask indicating availability
      - nearest predator's relative position + relative velocity

    Partial observability is preserved by masking out features beyond
    `sense_radius` (zeroing relative positions for things the fish can't see).
    """
    # -- neighbors --
    # (N, N, 2) pairwise deltas
    deltas = state.fish_pos[None, :, :] - state.fish_pos[:, None, :]
    sq_dists = jnp.sum(deltas * deltas, axis=-1)  # (N, N)

    # mask dead fish + self with a huge distance so they fall out of top-k
    big = jnp.float32(1e9)
    alive_mask = state.fish_alive[None, :]  # (1, N)
    self_mask = jnp.eye(cfg.n_fish, dtype=jnp.bool_)
    sq_dists = jnp.where(alive_mask, sq_dists, big)
    sq_dists = jnp.where(self_mask, big, sq_dists)

    # top-k nearest (negate for smallest-k)
    _, idx = jax.lax.top_k(-sq_dists, cfg.k_neighbors)  # (N, k)

    # gather neighbor positions / velocities
    nbr_pos = state.fish_pos[idx]  # (N, k, 2)
    nbr_vel = state.fish_vel[idx]  # (N, k, 2)

    rel_nbr_pos = nbr_pos - state.fish_pos[:, None, :]  # (N, k, 2)
    rel_nbr_vel = nbr_vel - state.fish_vel[:, None, :]

    # mask neighbors outside sense_radius → zero their features
    nbr_sq = jnp.sum(rel_nbr_pos * rel_nbr_pos, axis=-1)  # (N, k)
    in_range = (nbr_sq < cfg.sense_radius * cfg.sense_radius)[:, :, None]
    rel_nbr_pos = rel_nbr_pos * in_range
    rel_nbr_vel = rel_nbr_vel * in_range

    nbr_feat = jnp.concatenate([rel_nbr_pos, rel_nbr_vel], axis=-1)  # (N, k, 4)
    nbr_feat = nbr_feat.reshape(cfg.n_fish, -1)  # (N, 4k)

    # -- nearest n_food_obs food items (EXPANDED from 1 to configurable) --
    food_deltas = state.food_pos[None, :, :] - state.fish_pos[:, None, :]  # (N, M, 2)
    food_sq = jnp.sum(food_deltas * food_deltas, axis=-1)  # (N, M)
    food_sq = jnp.where(state.food_active[None, :], food_sq, big)
    # top-K nearest food (handle case where n_food < n_food_obs)
    k_food = min(cfg.n_food_obs, cfg.n_food)
    _, food_topk_idx = jax.lax.top_k(-food_sq, k_food)  # (N, k_food)

    food_topk_pos = state.food_pos[food_topk_idx]  # (N, k_food, 2)
    food_topk_rel = food_topk_pos - state.fish_pos[:, None, :]  # (N, k_food, 2)
    food_topk_sq = jnp.sum(food_topk_rel * food_topk_rel, axis=-1)  # (N, k_food)
    food_topk_visible = (food_topk_sq < cfg.sense_radius ** 2)[:, :, None]
    food_topk_rel = food_topk_rel * food_topk_visible

    food_feat = jnp.concatenate(
        [food_topk_rel, food_topk_visible.astype(jnp.float32)], axis=-1
    )  # (N, k_food, 3)
    # pad if n_food < n_food_obs
    if k_food < cfg.n_food_obs:
        pad = jnp.zeros((cfg.n_fish, cfg.n_food_obs - k_food, 3))
        food_feat = jnp.concatenate([food_feat, pad], axis=1)
    food_feat = food_feat.reshape(cfg.n_fish, -1)  # (N, 3*n_food_obs)

    # -- nearest predator --
    if cfg.n_predators > 0:
        pred_deltas = state.pred_pos[None, :, :] - state.fish_pos[:, None, :]
        pred_sq = jnp.sum(pred_deltas * pred_deltas, axis=-1)
        pred_idx = jnp.argmin(pred_sq, axis=-1)
        nearest_pred_pos = state.pred_pos[pred_idx]
        nearest_pred_vel = state.pred_vel[pred_idx]
        pred_rel_pos = nearest_pred_pos - state.fish_pos
        pred_rel_vel = nearest_pred_vel - state.fish_vel
        pred_in_range = (jnp.sum(pred_rel_pos * pred_rel_pos, axis=-1) < cfg.sense_radius ** 2)
        pred_rel_pos = pred_rel_pos * pred_in_range[:, None]
        pred_rel_vel = pred_rel_vel * pred_in_range[:, None]
        pred_feat = jnp.concatenate([pred_rel_pos, pred_rel_vel], axis=-1)
    else:
        pred_feat = jnp.zeros((cfg.n_fish, 4))

    # -- own position (normalized to [0, 1]) + velocity --
    own_pos = state.fish_pos / cfg.world_size  # (N, 2)
    own_vel = state.fish_vel  # (N, 2)

    parts = [own_pos, own_vel, nbr_feat, food_feat]
    if cfg.n_predators > 0:
        parts.append(pred_feat)
    obs = jnp.concatenate(parts, axis=-1)
    return obs


# =============================================================================
# Physics step
# =============================================================================


def _integrate(pos: jnp.ndarray, vel: jnp.ndarray, accel: jnp.ndarray,
               max_speed: float, drag: float, dt: float,
               world_size: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Semi-implicit Euler with linear drag + speed clamp + toroidal wrap."""
    vel = vel * (1.0 - drag) + accel * dt
    speed = jnp.linalg.norm(vel, axis=-1, keepdims=True) + 1e-8
    clamp = jnp.minimum(speed, max_speed) / speed
    vel = vel * clamp
    pos = pos + vel * dt
    # wall bounce — fish bounce off edges instead of wrapping through
    out_low = pos < 0
    out_high = pos > world_size
    pos = jnp.clip(pos, 0.0, world_size)
    vel = jnp.where(out_low | out_high, -vel * 0.5, vel)  # bounce with 50% damping
    return pos, vel


def _resolve_collisions(
    pos: jnp.ndarray,      # (N, 2)
    alive: jnp.ndarray,    # (N,) bool — inactive bodies don't collide
    radius: float,
    n_iter: int,
    world_size: float,
) -> jnp.ndarray:
    """Position-based dynamics (PBD) pairwise non-penetration for circles.

    Standard game-engine collision resolution: iterate a few relaxation
    passes, pushing overlapping body centers apart along the contact
    normal. Complexity is O(N^2) per iteration (N=few dozen → trivial on
    GPU via jax vmap over the pair matrix). No velocity update here —
    positions only — so the swarm stays physically consistent with JAX's
    functional step semantics. Verlet-style behavior emerges naturally
    because the next step's velocity is inferred from position delta over
    dt.

    The minimum allowed center-to-center distance is 2*radius (two bodies
    just touching). Toroidal wrapping is ignored for collision resolution:
    we use straight-line deltas. For a swarm that's rarely dense at the
    world-edge this is fine; for dense wrap-crossing contacts, add
    nearest-image deltas.
    """
    min_d = 2.0 * radius
    eps = 1e-8
    N = pos.shape[0]
    self_mask = jnp.eye(N, dtype=jnp.bool_)

    def _one_pass(pos):
        # (N, N, 2) pairwise delta  (from j to i)
        delta = pos[:, None, :] - pos[None, :, :]
        dist = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + eps)  # (N, N)
        overlap = jnp.maximum(0.0, min_d - dist)                # (N, N)

        # only real collisions: both bodies alive, different fish
        alive_pair = alive[:, None] & alive[None, :]
        valid = alive_pair & ~self_mask
        overlap = jnp.where(valid, overlap, 0.0)

        # unit contact normal (j → i)
        unit = delta / dist[..., None]

        # each body moves half the overlap away from each contact
        pushes = unit * (overlap * 0.5)[..., None]  # (N, N, 2)
        correction = jnp.sum(pushes, axis=1)        # (N, 2)
        return pos + correction

    for _ in range(n_iter):
        pos = _one_pass(pos)
    return jnp.clip(pos, 0.0, world_size)  # bounded walls, not toroidal


def _predator_action(state: EnvState, cfg: EnvConfig) -> jnp.ndarray:
    """Heuristic predators: steer toward nearest alive fish."""
    # (P, N, 2)
    deltas = state.fish_pos[None, :, :] - state.pred_pos[:, None, :]
    sq = jnp.sum(deltas * deltas, axis=-1)  # (P, N)
    sq = jnp.where(state.fish_alive[None, :], sq, 1e9)
    target_idx = jnp.argmin(sq, axis=-1)  # (P,)
    target_pos = state.fish_pos[target_idx]
    vec = target_pos - state.pred_pos
    norm = jnp.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8
    accel = (vec / norm) * cfg.pred_chase_gain * cfg.pred_max_speed
    return accel


def _whirlpool_accel(pos: jnp.ndarray, cfg: EnvConfig) -> jnp.ndarray:
    """Rotational + inward pull force from a whirlpool. (N, 2) -> (N, 2)"""
    center = jnp.array([cfg.whirlpool_x, cfg.whirlpool_y])
    d = center - pos                                        # (N, 2) toward center
    r = jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-8   # (N, 1)
    radial = d / r                                           # unit toward center
    tangent = jnp.stack([-radial[:, 1], radial[:, 0]], axis=-1)  # perpendicular (CCW)

    # linear falloff: full strength at center, zero at whirlpool_radius
    falloff = jnp.maximum(0.0, 1.0 - r / cfg.whirlpool_radius)  # (N, 1)
    accel = falloff * (cfg.whirlpool_strength * tangent + cfg.whirlpool_pull * radial)
    return accel


def step(
    state: EnvState,
    action: jnp.ndarray,
    key: jax.Array,
    cfg: EnvConfig,
) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    """Advance the env one tick.

    action: (N_fish, 2) desired accelerations, clipped internally
    returns:
        next_state, rewards (N_fish,), done scalar
    """
    # -- fish physics --
    # clip accel to safety
    accel_norm = jnp.linalg.norm(action, axis=-1, keepdims=True) + 1e-8
    accel_clip = jnp.minimum(accel_norm, cfg.fish_max_accel) / accel_norm
    fish_accel = action * accel_clip

    # whirlpool perturbation (test-time only)
    fish_accel = jnp.where(
        cfg.whirlpool_active,
        fish_accel + _whirlpool_accel(state.fish_pos, cfg),
        fish_accel,
    )

    # current perturbation (test-time only — constant directional force)
    fish_accel = jnp.where(
        cfg.current_active,
        fish_accel + jnp.array([cfg.current_dx, cfg.current_dy]),
        fish_accel,
    )

    # only alive fish move
    fish_accel = fish_accel * state.fish_alive[:, None]

    fish_pos, fish_vel = _integrate(
        state.fish_pos, state.fish_vel, fish_accel,
        cfg.fish_max_speed, cfg.fish_drag, cfg.dt, cfg.world_size,
    )

    # -- fish-fish collision resolution (PBD) --
    fish_pos = _resolve_collisions(
        fish_pos, state.fish_alive, cfg.fish_body_radius,
        cfg.collision_iters, cfg.world_size,
    )

    # -- predator physics --
    pred_accel = _predator_action(state, cfg)
    pred_pos, pred_vel = _integrate(
        state.pred_pos, state.pred_vel, pred_accel,
        cfg.pred_max_speed, cfg.pred_drag, cfg.dt, cfg.world_size,
    )

    # -- predator-predator collision --
    pred_alive = jnp.ones((cfg.n_predators,), dtype=jnp.bool_)
    pred_pos = _resolve_collisions(
        pred_pos, pred_alive, cfg.pred_body_radius,
        cfg.collision_iters, cfg.world_size,
    )

    # -- predation check --
    # (P, N, 2)
    kill_deltas = fish_pos[None, :, :] - pred_pos[:, None, :]
    kill_sq = jnp.sum(kill_deltas * kill_deltas, axis=-1)  # (P, N)
    killed_by_any = jnp.any(kill_sq < cfg.pred_kill_radius ** 2, axis=0)  # (N,)
    # only alive fish can be killed
    newly_dead = killed_by_any & state.fish_alive
    fish_alive = state.fish_alive & ~killed_by_any

    # -- food consumption --
    # (N, M, 2)
    food_deltas = state.food_pos[None, :, :] - fish_pos[:, None, :]
    food_sq = jnp.sum(food_deltas * food_deltas, axis=-1)  # (N, M)
    # a food patch is eaten if ANY alive fish is within eat_radius and it's active
    within = (food_sq < cfg.food_eat_radius ** 2) & fish_alive[:, None] & state.food_active[None, :]
    food_eaten = jnp.any(within, axis=0)  # (M,)
    # which fish eats it — first alive one (lowest idx) to break ties
    # for reward attribution: count food eaten per fish
    food_active_next = state.food_active & ~food_eaten

    # reward attribution: a fish gets +reward_food for each food in its own within[i,:]
    # (may double-count if multiple fish in same patch; cheap approximation)
    fish_food_count = jnp.sum(within & food_eaten[None, :], axis=-1)  # (N,)
    food_reward = fish_food_count.astype(jnp.float32) * cfg.reward_food

    # -- schooling bonus (small, for tier-0 readability) --
    # cohesion: mean distance to K nearest neighbors, penalty if too far
    # reuse part of observe logic for speed — cheap enough to recompute
    deltas2 = fish_pos[None, :, :] - fish_pos[:, None, :]
    sq2 = jnp.sum(deltas2 * deltas2, axis=-1)
    sq2 = jnp.where(jnp.eye(cfg.n_fish, dtype=jnp.bool_), 1e9, sq2)
    sq2 = jnp.where(fish_alive[None, :], sq2, 1e9)
    nearest_sq = jnp.min(sq2, axis=-1)  # (N,)
    # reward = 1 / (1 + dist) — bounded, smooth
    nearest_dist = jnp.sqrt(nearest_sq + 1e-6)
    cohesion_reward = cfg.reward_schooling / (1.0 + nearest_dist / cfg.sense_radius)

    # -- death penalty --
    death_reward = newly_dead.astype(jnp.float32) * cfg.reward_death

    time_penalty = jnp.full((cfg.n_fish,), cfg.reward_time_penalty)
    reward = (food_reward + cohesion_reward + death_reward + time_penalty) * state.fish_alive.astype(jnp.float32)

    # -- food respawn --
    key, k_resp = jax.random.split(key)
    respawn_mask = jax.random.uniform(k_resp, (cfg.n_food,)) < cfg.food_respawn_prob
    food_active_next = food_active_next | (respawn_mask & ~food_active_next)
    # move respawned food to fresh positions
    key, k_pos = jax.random.split(key)
    fresh_pos = jax.random.uniform(k_pos, (cfg.n_food, 2), minval=0.0, maxval=cfg.world_size)
    food_pos_next = jnp.where(
        (respawn_mask & ~state.food_active)[:, None], fresh_pos, state.food_pos
    )

    # -- reward history (sliding window, index 0 = most recent) --
    recent_shifted = jnp.concatenate(
        [reward[:, None], state.fish_recent_rewards[:, :-1]], axis=-1
    )

    # -- next state --
    t_next = state.t + 1
    done = t_next >= cfg.max_steps

    next_state = EnvState(
        fish_pos=fish_pos,
        fish_vel=fish_vel,
        fish_alive=fish_alive,
        fish_recent_rewards=recent_shifted,
        pred_pos=pred_pos,
        pred_vel=pred_vel,
        food_pos=food_pos_next,
        food_active=food_active_next,
        t=t_next,
        key=key,
    )
    return next_state, reward, done


# =============================================================================
# Vectorized batch API (for training)
# =============================================================================


def reset_batch(key: jax.Array, cfg: EnvConfig, batch_size: int) -> EnvState:
    keys = jax.random.split(key, batch_size)
    return jax.vmap(reset, in_axes=(0, None))(keys, cfg)


def step_batch(state: EnvState, action: jnp.ndarray, key: jax.Array,
               cfg: EnvConfig) -> tuple[EnvState, jnp.ndarray, jnp.ndarray]:
    """action: (B, N_fish, 2)."""
    batch_size = action.shape[0]
    keys = jax.random.split(key, batch_size)
    return jax.vmap(step, in_axes=(0, 0, 0, None))(state, action, keys, cfg)
