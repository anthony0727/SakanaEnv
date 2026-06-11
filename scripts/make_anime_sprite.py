"""
Generate an anime-style clownfish sprite procedurally via matplotlib + PIL.

The prod-mode renderer needs an actual anime-looking fish (not the monochrome
sakana logo). Rather than source a copyrighted asset, we draw a chibi clownfish
using primitive shapes: cell-shaded, saturated colors, thick outlines, big
expressive eye. Two color variants: default (orange) and leader (red).

Outputs:
    assets/anime/fish_default.png  — orange clownfish, facing +x
    assets/anime/fish_leader.png   — red clownfish, facing +x
    assets/anime/predator.png      — darker shark-like silhouette, facing +x

All sprites face +x at 0 rotation (head on the right). Renderers rotate
per-fish at draw time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
from PIL import Image


OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "anime"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_axes(fig, ax, path: Path):
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, transparent=True, dpi=200, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def make_clownfish(body_color: str, stripe_color: str = "#ffffff",
                   outline: str = "#1a1a1a", outline_lw: float = 3.5) -> Image.Image:
    """Draw a chibi clownfish facing +x. Return PIL RGBA image."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=200)

    # -- body: teardrop ellipse, slightly angled --
    body_verts = np.array([
        (1.0, 0.0),    # nose tip (right)
        (0.7, 0.42),   # upper forehead
        (0.2, 0.55),   # top of body
        (-0.4, 0.42),  # back hump
        (-0.65, 0.12), # start of tail attach
        (-0.75, -0.10),
        (-0.5, -0.4),  # belly
        (0.1, -0.55),  # bottom curve
        (0.65, -0.42), # lower jaw
        (1.0, 0.0),    # back to nose
    ])
    # smooth with Bezier — simple quadratic via CURVE3 codes
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * (len(body_verts) - 1)
    # For CURVE3 you need pairs (control, end), so adjust:
    # simpler: just LINETO — the outline is thick enough to look smooth-ish
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(body_verts) - 1)
    body = PathPatch(
        MplPath(body_verts, codes),
        facecolor=body_color, edgecolor=outline, linewidth=outline_lw, joinstyle="round",
    )
    ax.add_patch(body)

    # -- white stripes (clownfish markings) --
    # three vertical-ish curved stripes
    for (cx, width) in [(0.55, 0.12), (0.05, 0.14), (-0.35, 0.12)]:
        stripe_verts = np.array([
            (cx - width/2, 0.50),
            (cx + width/2, 0.52),
            (cx + width*0.7, -0.50),
            (cx - width*0.7, -0.52),
        ])
        stripe = mp.Polygon(
            stripe_verts, closed=True,
            facecolor=stripe_color, edgecolor=outline, linewidth=outline_lw * 0.6,
            joinstyle="round",
        )
        ax.add_patch(stripe)

    # -- tail fin (fork) --
    tail = mp.Polygon([
        (-0.65, 0.05),
        (-1.15, 0.55),
        (-1.25, 0.40),
        (-0.85, 0.0),
        (-1.25, -0.40),
        (-1.15, -0.55),
        (-0.65, -0.05),
    ], closed=True, facecolor=body_color, edgecolor=outline, linewidth=outline_lw, joinstyle="round")
    ax.add_patch(tail)

    # -- top fin --
    top_fin = mp.Polygon([
        (0.15, 0.52),
        (0.40, 0.78),
        (-0.15, 0.72),
        (-0.35, 0.48),
    ], closed=True, facecolor=body_color, edgecolor=outline, linewidth=outline_lw * 0.9,
        joinstyle="round")
    ax.add_patch(top_fin)

    # -- belly fin --
    belly_fin = mp.Polygon([
        (0.35, -0.45),
        (0.15, -0.70),
        (-0.05, -0.55),
    ], closed=True, facecolor=body_color, edgecolor=outline, linewidth=outline_lw * 0.9,
        joinstyle="round")
    ax.add_patch(belly_fin)

    # -- side fin (semicircle, gives depth) --
    side_fin = mp.Wedge(
        (0.25, -0.10), 0.22, 200, 350,
        facecolor=body_color, edgecolor=outline, linewidth=outline_lw * 0.8,
    )
    ax.add_patch(side_fin)

    # -- eye: big anime eye --
    eye_white = mp.Circle((0.70, 0.18), 0.13, facecolor="#ffffff",
                          edgecolor=outline, linewidth=outline_lw * 0.85)
    ax.add_patch(eye_white)
    pupil = mp.Circle((0.73, 0.16), 0.08, facecolor=outline)
    ax.add_patch(pupil)
    glint = mp.Circle((0.75, 0.21), 0.035, facecolor="#ffffff")
    ax.add_patch(glint)

    # -- mouth (small arc) --
    mouth = mp.FancyArrowPatch(
        (0.95, -0.08), (0.82, -0.18),
        connectionstyle="arc3,rad=0.3",
        arrowstyle="-", mutation_scale=1.0,
        color=outline, linewidth=outline_lw * 0.9,
    )
    ax.add_patch(mouth)

    _save_axes(fig, ax, OUT_DIR / "_tmp.png")
    img = Image.open(OUT_DIR / "_tmp.png").convert("RGBA")
    (OUT_DIR / "_tmp.png").unlink()
    # auto-crop to content
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def make_predator() -> Image.Image:
    """A larger darker silhouette — implied shark/big fish."""
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=200)
    outline = "#0a0a0a"

    # elongated angular body
    body_verts = np.array([
        (1.1, 0.0),
        (0.8, 0.35),
        (0.0, 0.45),
        (-0.9, 0.25),
        (-1.05, 0.05),
        (-1.05, -0.05),
        (-0.9, -0.25),
        (0.0, -0.45),
        (0.8, -0.35),
    ])
    body = mp.Polygon(body_verts, closed=True, facecolor="#333842",
                      edgecolor=outline, linewidth=4.5, joinstyle="round")
    ax.add_patch(body)

    # shark-like dorsal fin
    dorsal = mp.Polygon([
        (0.1, 0.42), (0.30, 0.80), (-0.25, 0.70),
    ], closed=True, facecolor="#333842", edgecolor=outline, linewidth=4)
    ax.add_patch(dorsal)

    # tail
    tail = mp.Polygon([
        (-0.95, 0.0), (-1.35, 0.45), (-1.45, 0.25),
        (-1.20, 0.0),
        (-1.45, -0.25), (-1.35, -0.45),
    ], closed=True, facecolor="#333842", edgecolor=outline, linewidth=4)
    ax.add_patch(tail)

    # pectoral fin
    pec = mp.Polygon([
        (0.2, -0.35), (0.05, -0.70), (-0.2, -0.45),
    ], closed=True, facecolor="#333842", edgecolor=outline, linewidth=3.5)
    ax.add_patch(pec)

    # gill marks
    for dx in [-0.05, 0.05, 0.15]:
        ax.plot([0.55 + dx, 0.45 + dx], [0.22, -0.22], color=outline, linewidth=2.5)

    # eye — menacing red
    eye = mp.Circle((0.80, 0.12), 0.08, facecolor="#d14b3a", edgecolor=outline, linewidth=3)
    ax.add_patch(eye)
    eye_pupil = mp.Circle((0.82, 0.11), 0.035, facecolor="#0a0a0a")
    ax.add_patch(eye_pupil)

    # mouth (toothy line)
    ax.plot([1.00, 0.68], [-0.05, -0.30], color=outline, linewidth=4)

    _save_axes(fig, ax, OUT_DIR / "_tmp.png")
    img = Image.open(OUT_DIR / "_tmp.png").convert("RGBA")
    (OUT_DIR / "_tmp.png").unlink()
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def main():
    print("generating anime sprites...")

    default_fish = make_clownfish(body_color="#ff7a3a")  # clownfish orange
    default_fish.save(OUT_DIR / "fish_default.png")
    print(f"  fish_default.png  {default_fish.size}")

    leader_fish = make_clownfish(body_color="#e0352a")   # red leader
    leader_fish.save(OUT_DIR / "fish_leader.png")
    print(f"  fish_leader.png   {leader_fish.size}")

    predator = make_predator()
    predator.save(OUT_DIR / "predator.png")
    print(f"  predator.png      {predator.size}")

    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
