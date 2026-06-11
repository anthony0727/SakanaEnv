"""
Embodied env wrapper for SakanaEnv — DreamerV3 interface.

Maps SakanaEnv's JAX state to Hafner's embodied.Env protocol:
  obs_space: dict of elements.Space (vector obs + reward + is_first/is_last/is_terminal)
  act_space: dict with 'action' (continuous 2D) + 'reset' (bool)
  step(action) -> dict of numpy arrays
"""

from __future__ import annotations

import elements
import embodied
import numpy as np

import jax
import jax.numpy as jnp

from .env import EnvConfig, EnvState, reset, step, observe, obs_dim
from .boids import boids_action


class SakanaEmbodiedEnv(embodied.Env):

    def __init__(self, fish_idx: int = 0, cfg: EnvConfig | None = None, seed: int = 0):
        self.cfg = cfg or EnvConfig(
            n_fish=3, n_predators=0, n_food=32,
            world_size=30.0, max_steps=500,
            fish_max_speed=2.0, fish_max_accel=4.0,
            sense_radius=50.0, k_neighbors=2,
            food_eat_radius=2.5, food_respawn_prob=0.15,
        )
        self.fish_idx = fish_idx
        self._key = jax.random.PRNGKey(seed)
        self._state = None
        self._step_count = 0
        self._od = obs_dim(self.cfg)

    @property
    def obs_space(self):
        return {
            'vector': elements.Space(np.float32, (self._od,)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.float32, (2,), -1.0, 1.0),
            'reset': elements.Space(bool),
        }

    def step(self, action):
        if action['reset'] or self._state is None:
            self._key, k = jax.random.split(self._key)
            self._state = reset(k, self.cfg)
            self._step_count = 0
            obs = self._get_obs()
            return {
                'vector': obs,
                'reward': np.float32(0.0),
                'is_first': True,
                'is_last': False,
                'is_terminal': False,
            }

        # scale action from [-1, 1] to [-max_accel, max_accel]
        act = np.asarray(action['action'], dtype=np.float32) * self.cfg.fish_max_accel

        # other fish use boids
        all_actions = np.array(boids_action(self._state, self.cfg))
        all_actions[self.fish_idx] = act

        self._key, k = jax.random.split(self._key)
        self._state, rewards, done = step(
            self._state, jnp.array(all_actions), k, self.cfg
        )
        self._step_count += 1

        obs = self._get_obs()
        reward = float(rewards[self.fish_idx])
        terminated = bool(not self._state.fish_alive[self.fish_idx])
        truncated = self._step_count >= self.cfg.max_steps
        is_last = terminated or truncated

        return {
            'vector': obs,
            'reward': np.float32(reward),
            'is_first': False,
            'is_last': is_last,
            'is_terminal': terminated,
        }

    def _get_obs(self) -> np.ndarray:
        all_obs = observe(self._state, self.cfg)
        return np.array(all_obs[self.fish_idx], dtype=np.float32)
