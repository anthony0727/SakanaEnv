# Evaluation Notes

The copied evaluation summary compares fixed learned policies against a decision-time planner in three environment modes.

Summary from the original public-safe eval text:

| Mode | PPO | DQN | TreeQuest-style AB-MCTS |
|---|---:|---:|---:|
| Normal | 711.8 +/- 38.6 | 214.8 +/- 24.9 | 927.0 +/- 44.1 |
| Whirlpool | 498.8 +/- 91.7 | 116.9 +/- 26.1 | 719.9 +/- 125.8 |
| Current | -4.6 +/- 5.1 | 6.3 +/- 9.0 | 589.4 +/- 31.8 |

The full table corresponds to:

```bash
uv run --extra eval python scripts/eval_treequest.py \
  --episodes 20 --steps 10000 --workers 8
```

For a quick dependency and checkpoint smoke test:

```bash
uv run --extra eval python scripts/eval_treequest.py \
  --episodes 1 --steps 200 --workers 1 --save-trajectories 0 \
  --out-dir /tmp/sakanaenv_eval_smoke
```

Interpretation:

- PPO and DQN use saved training checkpoints from `assets/weights/`.
- PPO uses the saved VecNormalize observation statistics.
- DQN uses the same 16-action discretization table used during training.
- The planner spends compute at decision time over short action sequences.
- Scores are cumulative per-fish environment reward over the shared competitive rollout.

Caveats:

- This is not equal-compute RL benchmarking.
- This is not a trained multi-agent PPO baseline.
- The planner includes heuristic proposal structure; do not describe it as pure random search.
- The result is best framed as robustness under perturbation: a fixed policy trained in calm water degrades under current/whirlpool perturbations, while decision-time search adapts.
- The full run is expensive because the planner spends compute at every decision step. Use the smoke test for dependency checks; use the full command only when regenerating the reported table.
