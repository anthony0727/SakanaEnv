import jax
import jax.numpy as jnp

from sakana_env import boids, env


def test_reset_observe_step_shapes():
    cfg = env.EnvConfig(
        n_fish=4,
        n_predators=1,
        n_food=8,
        k_neighbors=2,
        max_steps=8,
    )
    key = jax.random.PRNGKey(0)
    state = env.reset(key, cfg)

    obs = env.observe(state, cfg)
    assert obs.shape == (cfg.n_fish, env.obs_dim(cfg))

    actions = boids.boids_action(state, cfg)
    assert actions.shape == (cfg.n_fish, 2)

    key, step_key = jax.random.split(key)
    next_state, rewards, done = env.step(state, actions, step_key, cfg)
    assert next_state.fish_pos.shape == (cfg.n_fish, 2)
    assert rewards.shape == (cfg.n_fish,)
    assert jnp.asarray(done).shape == ()
