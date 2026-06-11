"""
Gymnasium wrapper for SakanaEnv — single-agent training interface.

Follows your marlenv pattern: wraps one fish for training while others
run boids. Compatible with Stable-Baselines3, PureJaxRL, CleanRL, or
any Gymnasium-compatible RL library.

Usage:
    from sakana_env.gym_wrapper import SakanaGymEnv
    env = SakanaGymEnv(fish_idx=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

    # SB3:
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import jax
import jax.numpy as jnp

from .env import EnvConfig, EnvState, reset, step, observe, obs_dim
from .boids import boids_action


def register_envs():
    from gymnasium.envs.registration import register
    register(
        id="SakanaEnv-v0",
        entry_point="sakana_env.gym_wrapper:SakanaGymEnv",
    )


class SakanaGymEnv(gym.Env):
    """Single-agent Gymnasium wrapper for SakanaEnv.

    Controls one fish (fish_idx). All other fish run boids policy.
    This is the same pattern as marlenv's SingleAgent wrapper.

    Observation: (obs_dim,) float32 — local neighborhood sensing
    Action: (2,) float32 — 2D acceleration, clipped to fish_max_accel
    Reward: scalar — food eaten + schooling bonus
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        fish_idx: int = 0,
        cfg: EnvConfig | None = None,
        seed: int = 0,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = cfg or EnvConfig(
            n_fish=3,
            n_predators=0,
            n_food=32,
            world_size=30.0,
            max_steps=500,
            fish_max_speed=2.0,
            fish_max_accel=4.0,
            sense_radius=50.0,  # > world diagonal → full food visibility
            k_neighbors=2,
            food_eat_radius=2.5,
            food_respawn_prob=0.15,
        )
        self.fish_idx = fish_idx
        self.render_mode = render_mode
        self._key = jax.random.PRNGKey(seed)
        self._state: EnvState | None = None
        self._step_count = 0

        od = obs_dim(self.cfg)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(od,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-self.cfg.fish_max_accel,
            high=self.cfg.fish_max_accel,
            shape=(2,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, k = jax.random.split(self._key)
        self._state = reset(k, self.cfg)
        self._step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        # build full action array: this fish uses the given action, others use boids
        all_actions = np.array(boids_action(self._state, self.cfg))
        all_actions[self.fish_idx] = action

        self._key, k = jax.random.split(self._key)
        self._state, rewards, done = step(
            self._state, jnp.array(all_actions), k, self.cfg
        )
        self._step_count += 1

        obs = self._get_obs()
        reward = float(rewards[self.fish_idx])
        terminated = bool(not self._state.fish_alive[self.fish_idx])
        truncated = self._step_count >= self.cfg.max_steps

        return obs, reward, terminated, truncated, {"food_total": reward}

    def _get_obs(self) -> np.ndarray:
        all_obs = observe(self._state, self.cfg)
        return np.array(all_obs[self.fish_idx], dtype=np.float32)


class SakanaMultiAgentEnv(gym.Env):
    """Multi-agent wrapper — all fish controlled externally.

    Action: (n_fish, 2) float32
    Observation: (n_fish, obs_dim) float32
    Reward: (n_fish,) float32

    For competitive evaluation: each fish slot gets a different trained policy.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: EnvConfig | None = None, seed: int = 0):
        super().__init__()
        self.cfg = cfg or EnvConfig(
            n_fish=3, n_predators=0, n_food=32,
            world_size=30.0, max_steps=500,
            fish_max_speed=2.0, fish_max_accel=4.0,
            sense_radius=50.0, k_neighbors=2,
            food_eat_radius=2.5, food_respawn_prob=0.15,
        )
        self._key = jax.random.PRNGKey(seed)
        self._state = None
        self._step_count = 0

        od = obs_dim(self.cfg)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.n_fish, od), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-self.cfg.fish_max_accel,
            high=self.cfg.fish_max_accel,
            shape=(self.cfg.n_fish, 2), dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, k = jax.random.split(self._key)
        self._state = reset(k, self.cfg)
        self._step_count = 0
        obs = np.array(observe(self._state, self.cfg), dtype=np.float32)
        return obs, {}

    def step(self, actions: np.ndarray):
        self._key, k = jax.random.split(self._key)
        self._state, rewards, done = step(
            self._state, jnp.array(actions), k, self.cfg
        )
        self._step_count += 1

        obs = np.array(observe(self._state, self.cfg), dtype=np.float32)
        rewards_np = np.array(rewards, dtype=np.float32)
        terminated = not bool(self._state.fish_alive.any())
        truncated = self._step_count >= self.cfg.max_steps

        return obs, rewards_np, terminated, truncated, {}

    @property
    def state(self) -> EnvState:
        return self._state
