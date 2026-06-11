"""
Dev-mode renderer for SakanaEnv.

Uses local generated fish sprites from /assets/anime/ as image markers,
rotated per-fish by velocity angle.
Sprites are pre-rotated into 72 angle bins at import time so per-frame render
is just a lookup + matplotlib AnnotationBbox placement.

Renderer-agnostic core contract:
    Anything that can consume a list[EnvState] + EnvConfig can render.
    matplotlib is the dev renderer; Three.js or Blender will be added as
    `viz_web.py` and `viz_blender.py` without touching env.py.

Usage:
    from sakana_env import env, viz
    states = [...]  # list of EnvState across a rollout
    viz.render_gif(states, cfg, "assets/rollout.gif")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle
import numpy as np
from PIL import Image

from .env import EnvConfig, EnvState


# -----------------------------------------------------------------------------
# Sprite loading + pre-rotation cache
# -----------------------------------------------------------------------------

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "anime"

_BG_COLOR = "#fdfdfa"      # off-white rice-paper
_FOOD_COLOR = "#b0b0b0"    # pale gray food dots
_DEAD_TINT = (207, 207, 207)
_PRED_TINT = (74, 74, 74)

_N_ROT_BINS = 72  # 5° resolution per bin


def _load_sprite(name: str) -> Image.Image:
    return Image.open(_ASSETS_DIR / name).convert("RGBA")


def _tint(img: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """Recolor all non-transparent 'ink' pixels to `color`. Preserves alpha."""
    arr = np.array(img)
    # "ink" = pixels that are opaque AND not already white/pale
    opaque = arr[..., 3] > 32
    darkish = arr[..., :3].mean(axis=-1) < 240
    mask = opaque & darkish
    out = arr.copy()
    out[mask, 0] = color[0]
    out[mask, 1] = color[1]
    out[mask, 2] = color[2]
    return Image.fromarray(out)


def _prerotate(img: Image.Image, n_bins: int) -> list[Image.Image]:
    """Rotate `img` into `n_bins` angle steps, counter-clockwise."""
    return [
        img.rotate(i * 360.0 / n_bins, resample=Image.BICUBIC, expand=True)
        for i in range(n_bins)
    ]


# Load the base sprites once.
_BLACK_SRC = _load_sprite("fish_default.png")
_RED_SRC = _load_sprite("fish_leader.png")
_DEAD_SRC = _tint(_BLACK_SRC, _DEAD_TINT)
_PRED_SRC = _load_sprite("predator.png")

# Pre-rotated caches — index by angle bin
_BLACK_ROT = _prerotate(_BLACK_SRC, _N_ROT_BINS)
_RED_ROT = _prerotate(_RED_SRC, _N_ROT_BINS)
_DEAD_ROT = _prerotate(_DEAD_SRC, _N_ROT_BINS)
_PRED_ROT = _prerotate(_PRED_SRC, _N_ROT_BINS)


def _sprite_for_angle(cache: list[Image.Image], angle_rad: float) -> np.ndarray:
    """Look up the pre-rotated sprite closest to `angle_rad` (math convention)."""
    deg = (np.degrees(angle_rad)) % 360.0
    idx = int(round(deg / (360.0 / _N_ROT_BINS))) % _N_ROT_BINS
    return np.asarray(cache[idx])


# -----------------------------------------------------------------------------
# Frame drawing
# -----------------------------------------------------------------------------


def _setup_axes(ax, cfg: EnvConfig) -> None:
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.set_aspect("equal")
    ax.set_facecolor(_BG_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _place_sprite(ax, sprite: np.ndarray, x: float, y: float, zoom: float) -> None:
    im = OffsetImage(sprite, zoom=zoom, interpolation="nearest")
    ab = AnnotationBbox(
        im, (x, y),
        frameon=False, pad=0.0,
        box_alignment=(0.5, 0.5),
        bboxprops=dict(edgecolor="none", facecolor="none", linewidth=0),
    )
    ax.add_artist(ab)


def draw_frame(
    ax,
    state: EnvState,
    cfg: EnvConfig,
    *,
    leader_idx: int | None = None,
    fish_zoom: float = 0.09,
    pred_zoom: float = 0.17,
) -> None:
    """Render one frame into an axes. Clears `ax` first."""
    ax.clear()
    _setup_axes(ax, cfg)

    pos = np.asarray(state.fish_pos)
    vel = np.asarray(state.fish_vel)
    alive = np.asarray(state.fish_alive)

    # food — lowest z-order
    food_pos = np.asarray(state.food_pos)
    food_active = np.asarray(state.food_active)
    for (fx, fy), active in zip(food_pos, food_active):
        if active:
            ax.add_patch(Circle((fx, fy), 0.45, color=_FOOD_COLOR, zorder=1))

    # fish
    for i in range(cfg.n_fish):
        ang = float(np.arctan2(vel[i, 1], vel[i, 0]))
        if not alive[i]:
            cache = _DEAD_ROT
        elif leader_idx is not None and i == leader_idx:
            cache = _RED_ROT
        else:
            cache = _BLACK_ROT
        sprite = _sprite_for_angle(cache, ang)
        _place_sprite(ax, sprite, float(pos[i, 0]), float(pos[i, 1]), fish_zoom)

    # predators
    pred_pos = np.asarray(state.pred_pos)
    pred_vel = np.asarray(state.pred_vel)
    for i in range(cfg.n_predators):
        ang = float(np.arctan2(pred_vel[i, 1], pred_vel[i, 0]))
        sprite = _sprite_for_angle(_PRED_ROT, ang)
        _place_sprite(ax, sprite, float(pred_pos[i, 0]), float(pred_pos[i, 1]), pred_zoom)


# -----------------------------------------------------------------------------
# Exporters
# -----------------------------------------------------------------------------


def render_frame(
    state: EnvState,
    cfg: EnvConfig,
    out_path: str,
    *,
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 150,
    leader_idx: int | None = None,
) -> None:
    """Save a single static frame as PNG or SVG."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    draw_frame(ax, state, cfg, leader_idx=leader_idx)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1, facecolor=_BG_COLOR)
    plt.close(fig)


def render_gif(
    states: list[EnvState],
    cfg: EnvConfig,
    out_path: str,
    *,
    fps: int = 24,
    figsize: tuple[float, float] = (7, 7),
    dpi: int = 100,
    leader_fn=None,
) -> None:
    """Animate a trajectory of states into a GIF.

    Args:
        states: sequence of EnvState snapshots (one per simulation tick)
        cfg: EnvConfig used for the rollout
        out_path: output GIF path (created if missing)
        fps: animation rate
        figsize: matplotlib figsize in inches
        dpi: matplotlib DPI
        leader_fn: optional (state, frame_idx) -> int | None; the returned
            index is rendered in red (the sakana logo's leader color). Use
            this to visualize, e.g., the argmax of the AB-MCTS-M posterior
            over fish-as-groups at each frame.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def _draw(i):
        leader = leader_fn(states[i], i) if leader_fn is not None else None
        draw_frame(ax, states[i], cfg, leader_idx=leader)
        return []

    anim = animation.FuncAnimation(
        fig, _draw, frames=len(states), interval=1000 / fps, blit=False
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(
        out_path,
        writer="pillow",
        fps=fps,
        dpi=dpi,
        savefig_kwargs={"facecolor": _BG_COLOR},
    )
    plt.close(fig)
