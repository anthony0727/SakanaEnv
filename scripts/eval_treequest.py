#!/usr/bin/env python
"""Competitive SakanaEnv eval using official TreeQuest AB-MCTS.

The rollout is:

* fish 0: saved PPO checkpoint;
* fish 1: saved DQN checkpoint;
* fish 2: TreeQuest AB-MCTS over short action sequences.

Outputs are written under `logs/treequest_eval/<timestamp>/`, with the
first trajectory for each condition also mirrored to `web/rollouts/` for
the Three.js renderer.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from sakana_env import viz_web
from sakana_env.env import EnvConfig, observe, reset, step
from sakana_env.sb3_weights import PolicyBundle
from sakana_env.treequest_action import TreeQuestActionConfig, plan_action_treequest


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "assets" / "weights"


@dataclass(frozen=True)
class EpisodeSpec:
    condition: str
    episode: int
    steps: int
    seed: int
    weights_dir: str
    out_dir: str
    mirror_web: bool
    save_trajectory: bool
    log_every: int
    tq_config: dict


def _condition_config(condition: str, steps: int) -> EnvConfig:
    common = dict(
        n_fish=3,
        n_predators=0,
        n_food=32,
        world_size=30.0,
        fish_max_speed=2.0,
        fish_max_accel=4.0,
        sense_radius=50.0,
        k_neighbors=2,
        food_eat_radius=2.5,
        food_respawn_prob=0.15,
        max_steps=steps,
    )
    if condition == "normal":
        return EnvConfig(**common)
    if condition == "whirlpool":
        return EnvConfig(
            **common,
            whirlpool_active=True,
            whirlpool_x=15.0,
            whirlpool_y=15.0,
            whirlpool_strength=5.0,
            whirlpool_pull=2.5,
            whirlpool_radius=14.0,
        )
    if condition == "current":
        return EnvConfig(
            **common,
            current_active=True,
            current_dx=3.0,
            current_dy=1.0,
        )
    raise ValueError(f"Unknown condition: {condition}")


def _run_episode(spec: EpisodeSpec) -> dict:
    cfg = _condition_config(spec.condition, spec.steps)
    policies = PolicyBundle.from_weights_dir(spec.weights_dir)
    tq_cfg = TreeQuestActionConfig(**spec.tq_config)
    rng = np.random.default_rng(spec.seed)
    observe_fn = jax.jit(observe, static_argnums=(1,))
    step_fn = jax.jit(step, static_argnums=(3,))

    key = jax.random.PRNGKey(spec.seed)
    state = reset(key, cfg)
    # Compile the two environment kernels once per worker/condition.
    warm_obs = observe_fn(state, cfg)
    warm_actions = jnp.zeros((cfg.n_fish, 2), dtype=jnp.float32)
    key, warm_key = jax.random.split(key)
    warm_state, _, _ = step_fn(state, warm_actions, warm_key, cfg)
    jax.block_until_ready(warm_obs)
    jax.block_until_ready(warm_state.fish_pos)
    states = [state] if spec.save_trajectory else None
    scores = np.zeros(3, dtype=np.float64)
    best_scores = []
    t0 = time.time()
    heartbeat_dir = Path(spec.out_dir) / "heartbeats"
    heartbeat_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = heartbeat_dir / f"{spec.condition}_ep{spec.episode:03d}.json"

    def write_heartbeat(t: int) -> None:
        elapsed = max(time.time() - t0, 1e-6)
        payload = {
            "condition": spec.condition,
            "episode": spec.episode,
            "step": int(t),
            "steps": int(spec.steps),
            "pct": float(t / max(spec.steps, 1)),
            "steps_per_sec": float(t / elapsed),
            "scores": {
                "ppo": float(scores[0]),
                "dqn": float(scores[1]),
                "treequest_ab_mcts": float(scores[2]),
            },
        }
        tmp = heartbeat_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(heartbeat_path)

    write_heartbeat(0)
    for t in range(spec.steps):
        obs_jax = observe_fn(state, cfg)
        jax.block_until_ready(obs_jax)
        obs = np.asarray(obs_jax, dtype=np.float32)
        a0 = policies.ppo_fish0.predict(obs[0])
        a1 = policies.dqn_fish1.predict(obs[1])
        a2, tq_stats = plan_action_treequest(
            state,
            cfg,
            2,
            policies,
            rng,
            tq_cfg,
            step_fn=step_fn,
            observe_fn=observe_fn,
        )
        best_scores.append(tq_stats.best_score)

        actions = jnp.asarray(np.stack([a0, a1, a2]).astype(np.float32))
        key, subkey = jax.random.split(key)
        state, rewards, _ = step_fn(state, actions, subkey, cfg)
        jax.block_until_ready(state.fish_pos)
        scores += np.asarray(rewards[:3], dtype=np.float64)
        if states is not None:
            states.append(state)
        if spec.log_every > 0 and (t + 1) % spec.log_every == 0:
            write_heartbeat(t + 1)
    write_heartbeat(spec.steps)

    trajectory_path = None
    if states is not None:
        out_dir = Path(spec.out_dir)
        trajectory_path = out_dir / "rollouts" / f"{spec.condition}_ep{spec.episode:03d}_treequest.json"
        viz_web.export_trajectory(states, cfg, str(trajectory_path), leaders=[2] * len(states))
        if spec.mirror_web:
            web_path = ROOT / "web" / "rollouts" / f"{spec.condition}_treequest.json"
            viz_web.export_trajectory(states, cfg, str(web_path), leaders=[2] * len(states))

    return {
        "condition": spec.condition,
        "episode": spec.episode,
        "seed": spec.seed,
        "steps": spec.steps,
        "scores": {
            "ppo": float(scores[0]),
            "dqn": float(scores[1]),
            "treequest_ab_mcts": float(scores[2]),
        },
        "treequest_best_score_mean": float(np.mean(best_scores)) if best_scores else 0.0,
        "seconds": float(time.time() - t0),
        "trajectory_path": str(trajectory_path) if trajectory_path else None,
    }


def _summarize(results: list[dict]) -> dict:
    summary = {}
    for condition in sorted({r["condition"] for r in results}):
        rows = [r for r in results if r["condition"] == condition]
        condition_summary = {}
        for key in ["ppo", "dqn", "treequest_ab_mcts"]:
            values = np.array([r["scores"][key] for r in rows], dtype=np.float64)
            condition_summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "n": int(values.size),
            }
        summary[condition] = condition_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SakanaEnv with official TreeQuest AB-MCTS.")
    parser.add_argument("--conditions", nargs="+", default=["normal", "whirlpool", "current"], choices=["normal", "whirlpool", "current"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    parser.add_argument("--seed", type=int, default=727)
    parser.add_argument("--algorithm", default="A", choices=["A", "M", "ABMCTS-A", "ABMCTS-M"])
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--save-trajectories", type=int, default=1, help="Number of episodes per condition to export.")
    parser.add_argument("--log-every", type=int, default=250, help="Write per-episode heartbeat JSON every N env steps.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "logs" / "treequest_eval" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rollouts").mkdir(exist_ok=True)

    logging.basicConfig(
        filename=out_dir / "eval.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Starting TreeQuest eval: %s", vars(args))

    policy_probe = PolicyBundle.from_weights_dir(WEIGHTS_DIR)
    try:
        treequest_version = importlib.metadata.version("treequest")
    except Exception:
        treequest_version = "unknown"

    tq_config = asdict(TreeQuestActionConfig(
        algorithm=args.algorithm,
        budget=args.budget,
        batch_size=args.batch_size,
        horizon=args.horizon,
    ))
    specs = []
    for condition in args.conditions:
        for ep in range(args.episodes):
            specs.append(EpisodeSpec(
                condition=condition,
                episode=ep,
                steps=args.steps,
                seed=args.seed + 10_000 * args.conditions.index(condition) + ep,
                weights_dir=str(WEIGHTS_DIR),
                out_dir=str(out_dir),
                mirror_web=(ep == 0),
                save_trajectory=(ep < args.save_trajectories),
                log_every=args.log_every,
                tq_config=tq_config,
            ))

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_run_episode, spec) for spec in specs]
        with tqdm(total=len(futures), desc="TreeQuest eval", unit="episode") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                logging.info("Episode done: %s", result)
                pbar.set_postfix({
                    "condition": result["condition"],
                    "treequest": f"{result['scores']['treequest_ab_mcts']:.1f}",
                })
                pbar.update(1)

    results = sorted(results, key=lambda r: (r["condition"], r["episode"]))
    summary = _summarize(results)
    payload = {
        "args": vars(args),
        "provenance": {
            "treequest_version": treequest_version,
            "treequest_algorithm": "ABMCTSA" if str(args.algorithm).upper() in {"A", "ABMCTS-A"} else "ABMCTSM",
            "ppo_uses_vecnormalize": policy_probe.ppo_fish0.obs_normalizer is not None,
            "dqn_action_space": "16 discrete directions/magnitudes from SB3 DQN wrapper",
            "score_definition": "per-fish cumulative environment reward over the same competitive rollout",
        },
        "treequest_config": tq_config,
        "results": results,
        "summary": summary,
    }
    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return path.name

    for result in results:
        if result.get("trajectory_path"):
            result["trajectory_path"] = display_path(Path(result["trajectory_path"]))

    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    for condition, stats in summary.items():
        lines.append(f"{condition.upper()}: {args.episodes} episodes x {args.steps} steps")
        for key, label in [
            ("ppo", "PPO"),
            ("dqn", "DQN"),
            ("treequest_ab_mcts", f"TreeQuest AB-MCTS-{args.algorithm.upper()}"),
        ]:
            row = stats[key]
            lines.append(f"  {label:18s}: {row['mean']:8.1f} +/- {row['std']:.1f}")
        lines.append("")
    text = "\n".join(lines)
    (out_dir / "results.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"wrote: {display_path(out_dir)}")


if __name__ == "__main__":
    main()
