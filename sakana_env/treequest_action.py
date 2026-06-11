"""Official TreeQuest adapter for SakanaEnv action-sequence planning.

This file does not implement AB-MCTS itself. It adapts SakanaEnv to
TreeQuest by defining:

* node state: a short open-loop action sequence for one fish;
* GEN: create or refine an action sequence;
* score: normalized environment rollout reward for that sequence.

TreeQuest then owns the search algorithm via `treequest.ABMCTSA` or
`treequest.ABMCTSM`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .env import EnvConfig, EnvState, observe, step
from .sb3_weights import PolicyBundle


StepFn = Callable[[EnvState, jnp.ndarray, jax.Array, EnvConfig], tuple[EnvState, jnp.ndarray, jnp.ndarray]]
ObserveFn = Callable[[EnvState, EnvConfig], jnp.ndarray]


@dataclass(frozen=True)
class TreeQuestActionConfig:
    """Search budget and action-sequence parameters."""

    algorithm: str = "A"
    budget: int = 32
    batch_size: int = 1
    horizon: int = 8
    discount: float = 0.98
    noise_scale: float = 0.55
    terminal_food_bonus: float = 0.30
    score_temperature: float = 0.45


@dataclass(frozen=True)
class ActionPlan:
    """State object stored in each TreeQuest node."""

    actions: np.ndarray


@dataclass(frozen=True)
class TreeQuestStats:
    """Compact diagnostics from one TreeQuest planning call."""

    algorithm: str
    budget: int
    horizon: int
    best_score: float


def _clip_action(action: np.ndarray, max_accel: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    norm = float(np.linalg.norm(action))
    if norm > max_accel:
        action = action / (norm + 1e-8) * max_accel
    return action.astype(np.float32)


def _nearest_food_action(state: EnvState, cfg: EnvConfig, fish_idx: int) -> np.ndarray:
    fish_pos = np.asarray(state.fish_pos[fish_idx])
    food_pos = np.asarray(state.food_pos)
    active = np.asarray(state.food_active)
    if not np.any(active):
        return np.zeros(2, dtype=np.float32)
    deltas = food_pos - fish_pos[None, :]
    sq = np.sum(deltas * deltas, axis=-1)
    sq = np.where(active, sq, 1e9)
    direction = deltas[int(np.argmin(sq))]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return np.zeros(2, dtype=np.float32)
    return (direction / norm * cfg.fish_max_accel).astype(np.float32)


def _sequence_from_base(
    base: np.ndarray,
    rng: np.random.Generator,
    cfg: EnvConfig,
    tq_cfg: TreeQuestActionConfig,
) -> np.ndarray:
    seq = np.repeat(base[None, :], tq_cfg.horizon, axis=0)
    noise = rng.normal(0.0, cfg.fish_max_accel * tq_cfg.noise_scale, size=seq.shape)
    taper = np.linspace(1.0, 0.35, tq_cfg.horizon, dtype=np.float32)[:, None]
    seq = seq + noise * taper
    return np.stack([_clip_action(a, cfg.fish_max_accel) for a in seq]).astype(np.float32)


def _generate_root_plan(
    state: EnvState,
    cfg: EnvConfig,
    fish_idx: int,
    rng: np.random.Generator,
    tq_cfg: TreeQuestActionConfig,
) -> ActionPlan:
    food = _nearest_food_action(state, cfg, fish_idx)
    theta = rng.uniform(0.0, 2 * np.pi)
    random_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32) * cfg.fish_max_accel
    slow = food * 0.45
    choices = [food, random_dir, slow, np.zeros(2, dtype=np.float32)]
    probs = np.array([0.58, 0.27, 0.10, 0.05], dtype=np.float32)
    base = choices[int(rng.choice(len(choices), p=probs / probs.sum()))]
    return ActionPlan(_sequence_from_base(base, rng, cfg, tq_cfg))


def _refine_plan(
    parent: ActionPlan,
    state: EnvState,
    cfg: EnvConfig,
    fish_idx: int,
    rng: np.random.Generator,
    tq_cfg: TreeQuestActionConfig,
) -> ActionPlan:
    seq = parent.actions.copy()
    start = int(rng.integers(0, tq_cfg.horizon))
    width = int(rng.integers(1, max(2, tq_cfg.horizon - start + 1)))
    if rng.random() < 0.45:
        anchor = _nearest_food_action(state, cfg, fish_idx)
        seq[start : start + width] = _sequence_from_base(anchor, rng, cfg, tq_cfg)[start : start + width]
    else:
        noise = rng.normal(0.0, cfg.fish_max_accel * tq_cfg.noise_scale, size=(width, 2))
        taper = np.linspace(1.0, 0.4, width, dtype=np.float32)[:, None]
        seq[start : start + width] = seq[start : start + width] + noise * taper
    seq = np.stack([_clip_action(a, cfg.fish_max_accel) for a in seq]).astype(np.float32)
    return ActionPlan(seq)


def _baseline_actions(
    sim_state: EnvState,
    cfg: EnvConfig,
    policies: PolicyBundle,
    observe_fn: ObserveFn,
) -> np.ndarray:
    obs = np.asarray(observe_fn(sim_state, cfg), dtype=np.float32)
    actions = np.zeros((cfg.n_fish, 2), dtype=np.float32)
    actions[0] = policies.ppo_fish0.predict(obs[0])
    actions[1] = policies.dqn_fish1.predict(obs[1])
    return actions


def _terminal_food_proximity(state: EnvState, cfg: EnvConfig, fish_idx: int) -> float:
    active = np.asarray(state.food_active)
    if not np.any(active):
        return 0.0
    pos = np.asarray(state.fish_pos[fish_idx])
    food = np.asarray(state.food_pos)[active]
    nearest = float(np.linalg.norm(food - pos[None, :], axis=-1).min())
    return max(0.0, 1.0 - nearest / max(cfg.world_size, 1e-6))


def score_plan(
    state: EnvState,
    cfg: EnvConfig,
    fish_idx: int,
    plan: ActionPlan,
    policies: PolicyBundle,
    rng: np.random.Generator,
    tq_cfg: TreeQuestActionConfig,
    *,
    step_fn: StepFn = step,
    observe_fn: ObserveFn = observe,
) -> float:
    """Score a plan as normalized rollout reward in `[0, 1]`."""
    sim_state = state
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    raw = 0.0
    gamma = 1.0
    for action in plan.actions:
        actions = _baseline_actions(sim_state, cfg, policies, observe_fn)
        actions[fish_idx] = action
        key, subkey = jax.random.split(key)
        sim_state, rewards, _ = step_fn(sim_state, jnp.asarray(actions), subkey, cfg)
        raw += gamma * float(np.asarray(rewards)[fish_idx])
        gamma *= tq_cfg.discount
        if not bool(np.asarray(sim_state.fish_alive)[fish_idx]):
            break
    raw += tq_cfg.terminal_food_bonus * _terminal_food_proximity(sim_state, cfg, fish_idx)
    return float(1.0 / (1.0 + np.exp(-raw / max(tq_cfg.score_temperature, 1e-6))))


def _build_algo(tq, tq_cfg: TreeQuestActionConfig):
    algo_name = tq_cfg.algorithm.upper()
    if algo_name in {"A", "ABMCTS-A", "ABMCTSA"}:
        return tq.ABMCTSA()
    if algo_name in {"M", "ABMCTS-M", "ABMCTSM"}:
        try:
            return tq.ABMCTSM(max_process_workers=max(1, tq_cfg.batch_size))
        except TypeError:
            return tq.ABMCTSM()
    raise ValueError(f"Unknown TreeQuest algorithm: {tq_cfg.algorithm!r}")


def plan_action_treequest(
    state: EnvState,
    cfg: EnvConfig,
    fish_idx: int,
    policies: PolicyBundle,
    rng: np.random.Generator,
    tq_cfg: TreeQuestActionConfig | None = None,
    *,
    step_fn: StepFn = step,
    observe_fn: ObserveFn = observe,
) -> tuple[np.ndarray, TreeQuestStats]:
    """Run official TreeQuest and return the first action of the best plan."""
    try:
        import treequest as tq
    except ImportError as exc:
        raise RuntimeError(
            "TreeQuest is not installed. Run `uv sync --extra eval` for the "
            "saved-checkpoint eval, or `uv sync --extra planner` for planner-only use."
        ) from exc

    tq_cfg = tq_cfg or TreeQuestActionConfig()

    def generate(parent_state: ActionPlan | None) -> tuple[ActionPlan, float]:
        if parent_state is None:
            plan = _generate_root_plan(state, cfg, fish_idx, rng, tq_cfg)
        else:
            plan = _refine_plan(parent_state, state, cfg, fish_idx, rng, tq_cfg)
        return plan, score_plan(
            state,
            cfg,
            fish_idx,
            plan,
            policies,
            rng,
            tq_cfg,
            step_fn=step_fn,
            observe_fn=observe_fn,
        )

    algo = _build_algo(tq, tq_cfg)
    search_tree = algo.init_tree()
    generate_fns = {"action-sequence": generate}
    batch_size = max(1, int(tq_cfg.batch_size))

    if batch_size == 1 or not hasattr(algo, "ask_batch"):
        for _ in range(int(tq_cfg.budget)):
            search_tree = algo.step(search_tree, generate_fns)
    else:
        actions = list(generate_fns)
        expanded = 0
        while expanded < int(tq_cfg.budget):
            n = min(batch_size, int(tq_cfg.budget) - expanded)
            search_tree, trials = algo.ask_batch(search_tree, n, actions)
            for trial in trials:
                result = generate_fns[trial.action](trial.parent_state)
                search_tree = algo.tell(search_tree, trial.trial_id, result)
                expanded += 1

    best_plan, best_score = tq.top_k(search_tree, algo, k=1)[0]
    if best_plan is None:
        action = _nearest_food_action(state, cfg, fish_idx)
    else:
        action = best_plan.actions[0].astype(np.float32)
    return action, TreeQuestStats(
        algorithm=tq_cfg.algorithm,
        budget=int(tq_cfg.budget),
        horizon=int(tq_cfg.horizon),
        best_score=float(best_score),
    )


__all__ = [
    "ActionPlan",
    "TreeQuestActionConfig",
    "TreeQuestStats",
    "plan_action_treequest",
    "score_plan",
]
