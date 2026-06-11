"""
Web renderer — export a trajectory as JSON for a Three.js front-end.

This is the prod-mode bridge. The dev renderer (viz.py, matplotlib) is used
for iteration and debugging; the prod renderer is a Three.js scene loaded
from a static JSON trajectory.

Core contract kept identical to viz.py: anything that produces a list of
EnvState objects can drive any renderer. No physics or env logic lives in
the renderer.

This module only does the JSON export. The Three.js page that consumes it
lives at `web/` in the repo root (scaffolded separately).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import EnvConfig, EnvState


def export_trajectory(
    states: list[EnvState],
    cfg: EnvConfig,
    out_path: str,
    *,
    leaders: list[int | None] | None = None,
) -> None:
    """Serialize a rollout as JSON for the Three.js web renderer.

    Schema:
    {
      "cfg": { "n_fish": int, "n_predators": int, "n_food": int,
                "world_size": float, "dt": float },
      "frames": [
        {
          "t": int,
          "fish":      [[x, y, vx, vy, alive, is_leader], ...],
          "predators": [[x, y, vx, vy], ...],
          "food":      [[x, y, active], ...]
        },
        ...
      ]
    }

    Args:
        states: sequence of EnvState snapshots (one per simulation tick)
        cfg: EnvConfig used for the rollout
        out_path: output JSON path
        leaders: optional, per-frame index of the "leader" fish to render in red.
                 Pass `None` for a frame = no leader highlight.
    """
    payload = {
        "cfg": {
            "n_fish": int(cfg.n_fish),
            "n_predators": int(cfg.n_predators),
            "n_food": int(cfg.n_food),
            "world_size": float(cfg.world_size),
            "dt": float(cfg.dt),
        },
        "frames": [],
    }

    for t, state in enumerate(states):
        leader_idx = leaders[t] if leaders is not None else None
        fish_pos = np.asarray(state.fish_pos)
        fish_vel = np.asarray(state.fish_vel)
        fish_alive = np.asarray(state.fish_alive)

        fish_rows = []
        for i in range(cfg.n_fish):
            fish_rows.append([
                float(fish_pos[i, 0]),
                float(fish_pos[i, 1]),
                float(fish_vel[i, 0]),
                float(fish_vel[i, 1]),
                bool(fish_alive[i]),
                bool(leader_idx is not None and i == leader_idx),
            ])

        pred_pos = np.asarray(state.pred_pos)
        pred_vel = np.asarray(state.pred_vel)
        pred_rows = [
            [float(pred_pos[i, 0]), float(pred_pos[i, 1]),
             float(pred_vel[i, 0]), float(pred_vel[i, 1])]
            for i in range(cfg.n_predators)
        ]

        food_pos = np.asarray(state.food_pos)
        food_active = np.asarray(state.food_active)
        food_rows = [
            [float(food_pos[i, 0]), float(food_pos[i, 1]), bool(food_active[i])]
            for i in range(cfg.n_food)
        ]

        payload["frames"].append({
            "t": int(t),
            "fish": fish_rows,
            "predators": pred_rows,
            "food": food_rows,
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)


__all__ = ["export_trajectory"]
