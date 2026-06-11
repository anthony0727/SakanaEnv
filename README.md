# SakanaEnv

SakanaEnv is a JAX-native multi-agent swarm environment for testing decentralized coordination, robustness, and decision-time planning in an embodied setting.

The canonical API is functional JAX: `reset`, `observe`, and `step` operate on immutable PyTree state. A small Gymnasium adapter is included for compatibility with existing RL tooling, but the core environment is not forced into Gymnasium.

<video src="assets/sakanaenv-new-full-720p.m4v" controls muted loop playsinline width="720"></video>

[Open the 3D demo video](assets/sakanaenv-new-full-720p.m4v)

## Quickstart

```bash
uv sync
uv run python examples/run_boids.py
```

This runs a boids baseline and writes `assets/boids.gif`.

## JAX API

```python
import jax
from sakana_env import boids, env

cfg = env.EnvConfig(n_fish=64, n_predators=2, n_food=32)
key = jax.random.PRNGKey(0)
state = env.reset(key, cfg)

for _ in range(cfg.max_steps):
    obs = env.observe(state, cfg)
    action = boids.boids_action(state, cfg)
    key, step_key = jax.random.split(key)
    state, reward, done = env.step(state, action, step_key, cfg)
```

State fields are JAX arrays and the state object is a PyTree-compatible `NamedTuple`, so rollouts can be `jit`-compiled and batched.

## Web Demo

The repo includes a Three.js viewer plus staged rollout JSON and GLB assets:

```bash
uv run python -m http.server 8000 --directory web
```

Then open `http://localhost:8000`.

The web viewer uses the staged Meshy-style fish, onigiri, and sakura GLBs in
`web/rollouts/`. To regenerate only the lightweight boids rollout JSON:

```bash
uv run python scripts/make_web_demo.py --steps 240
```

That command does not overwrite staged GLBs. To intentionally replace them with
the procedural fish/predator assets, pass `--stage-procedural-assets
--force-assets`.

To capture the web demo as MP4:

```bash
uv run --extra video python scripts/capture_web_video.py --out artifacts/sakanaenv_demo.mp4
```

If Playwright has no browser installed yet, run:

```bash
uv run --extra video python -m playwright install chromium
```

## Gymnasium Adapter

Install the optional adapter dependencies:

```bash
uv sync --extra gym
```

```python
from sakana_env.gym_wrapper import SakanaGymEnv

gym_env = SakanaGymEnv(fish_idx=0)
obs, info = gym_env.reset(seed=0)
```

## Evaluation

The original artifact includes saved PPO/DQN weights and a TreeQuest-style action-sequence planning eval. See `docs/evaluation.md` for the copied summary, reproducibility commands, and caveats.

Fast smoke check:

```bash
uv run --extra eval python scripts/eval_treequest.py \
  --episodes 1 --steps 200 --workers 1 --save-trajectories 0 \
  --out-dir /tmp/sakanaenv_eval_smoke
```

Full reproduction run for the table:

```bash
uv run --extra eval python scripts/eval_treequest.py \
  --episodes 20 --steps 10000 --workers 8
```

## Assets

Included public demo assets:

- anime GIF and sprite assets
- procedural fish/predator GLBs
- web demo GLBs for fish, onigiri, and sakura
- staged rollout JSON for normal, whirlpool, and current conditions
- saved PPO/DQN weights used by the evaluation scripts

## License

Apache-2.0.
