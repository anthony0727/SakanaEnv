"""
Boids baseline — Reynolds 1987 rules as an action policy.

This is the non-learning baseline SakanaEnv is measured against. It takes
the env's own observation vector (each fish sees its K nearest neighbors
plus nearest food and nearest predator) and emits an acceleration action.

The three classic Reynolds rules:
    1. Separation   — avoid crowding local neighbors
    2. Alignment    — steer toward the average heading of local neighbors
    3. Cohesion     — steer toward the average position of local neighbors

Plus two extensions for this env:
    4. Food seek    — steer toward nearest visible food patch
    5. Predator flee — steer away from nearest visible predator (exponential urgency)

Every operation is vmap-friendly — runs at full env speed inside jax.jit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .env import EnvConfig, EnvState


def boids_action(
    state: EnvState,
    cfg: EnvConfig,
    *,
    w_separation: float = 2.0,
    w_alignment: float = 1.0,
    w_cohesion: float = 1.0,
    w_food: float = 1.5,
    w_flee: float = 4.0,
    separation_radius: float = 3.0,
) -> jnp.ndarray:
    """Reynolds boids → (N_fish, 2) acceleration vector.

    Uses `state` directly rather than precomputed observations because the
    boids rules are themselves the canonical sensing model — computing them
    from obs would require recovering raw positions anyway.
    """
    pos = state.fish_pos  # (N, 2)
    vel = state.fish_vel  # (N, 2)
    alive = state.fish_alive  # (N,)

    # -- pairwise deltas --
    # delta[i, j] = pos[j] - pos[i]
    delta = pos[None, :, :] - pos[:, None, :]  # (N, N, 2)
    dist_sq = jnp.sum(delta * delta, axis=-1)  # (N, N)

    # mask self + dead neighbors
    self_mask = jnp.eye(cfg.n_fish, dtype=jnp.bool_)
    alive_mask = alive[None, :] & ~self_mask  # (1, N) broadcast
    dist_sq_masked = jnp.where(alive_mask, dist_sq, 1e9)

    # sensing-radius gate
    sense_sq = cfg.sense_radius ** 2
    in_range = dist_sq_masked < sense_sq  # (N, N) bool
    # also exclude self / dead
    in_range = in_range & alive_mask

    # -- cohesion: steer toward mean neighbor position --
    # weighted by in_range mask
    mask_f = in_range.astype(jnp.float32)[:, :, None]  # (N, N, 1)
    n_neighbors = jnp.sum(in_range, axis=-1, keepdims=True) + 1e-6  # (N, 1)
    mean_pos = jnp.sum(pos[None, :, :] * mask_f, axis=1) / n_neighbors  # (N, 2)
    # desired: move toward (mean_pos - self_pos). zero if no neighbors.
    cohesion = (mean_pos - pos) * (n_neighbors > 1e-3)

    # -- alignment: steer toward mean neighbor velocity --
    mean_vel = jnp.sum(vel[None, :, :] * mask_f, axis=1) / n_neighbors  # (N, 2)
    alignment = (mean_vel - vel) * (n_neighbors > 1e-3)

    # -- separation: repel from VERY close neighbors --
    sep_sq = separation_radius ** 2
    too_close = (dist_sq_masked < sep_sq) & alive_mask  # (N, N)
    # invert direction: delta[i,j] = pos[j]-pos[i], so -delta pushes i away from j
    # weight by 1/dist so closer neighbors dominate
    inv_dist = 1.0 / (jnp.sqrt(dist_sq_masked) + 0.5)  # (N, N), 0.5 floor avoids nan
    sep_weight = too_close.astype(jnp.float32) * inv_dist  # (N, N)
    sep_vec = -jnp.sum(delta * sep_weight[:, :, None], axis=1)  # (N, 2)
    separation = sep_vec

    # -- food seek: steer toward nearest visible food patch --
    # (N, M, 2)
    food_delta = state.food_pos[None, :, :] - pos[:, None, :]
    food_sq = jnp.sum(food_delta * food_delta, axis=-1)
    food_sq = jnp.where(state.food_active[None, :], food_sq, 1e9)
    nearest_food_idx = jnp.argmin(food_sq, axis=-1)  # (N,)
    nearest_food_pos = state.food_pos[nearest_food_idx]
    food_vec = nearest_food_pos - pos
    food_visible = (jnp.sum(food_vec * food_vec, axis=-1) < sense_sq)
    food = food_vec * food_visible[:, None]

    # -- predator flee: exponential urgency at close range --
    if cfg.n_predators > 0:
        pred_delta = state.pred_pos[None, :, :] - pos[:, None, :]
        pred_sq = jnp.sum(pred_delta * pred_delta, axis=-1)
        nearest_pred_idx = jnp.argmin(pred_sq, axis=-1)
        nearest_pred_pos = state.pred_pos[nearest_pred_idx]
        pred_vec = pos - nearest_pred_pos
        pred_dist = jnp.linalg.norm(pred_vec, axis=-1, keepdims=True) + 1e-6
        pred_visible = (pred_dist[:, 0] < cfg.sense_radius)
        urgency = (1.0 / pred_dist) * pred_visible[:, None]
        flee = (pred_vec / pred_dist) * urgency * cfg.sense_radius
    else:
        flee = jnp.zeros_like(pos)

    # -- combine --
    accel = (
        w_separation * separation
        + w_alignment * alignment
        + w_cohesion * cohesion
        + w_food * food
        + w_flee * flee
    )

    # zero out dead fish
    accel = accel * alive[:, None]
    return accel
