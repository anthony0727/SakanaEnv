"""
Procedural 3D fish mesh generator via trimesh.

Same philosophy as unixpickle/model3d (compose solids, export mesh) but in
Python and targeting GLB for three.js consumption. The fish is built from
primitives: an elongated ellipsoid body, tapered tail, top/bottom/pectoral
fins, and eyes. Colors are baked into vertex colors on the mesh; the
three.js renderer applies `MeshToonMaterial` + inverted-hull outline on
top for the anime look.

Coordinate convention: the fish faces +x at 0 rotation (head on right).
Y is up. Z is lateral width. All scales are in world units compatible
with SakanaEnv's 80x80 world — a single fish is ~3 units long.

Export formats:
    GLB  — for three.js / gltf-loader / browser
    OBJ  — for Blender / debug
    PNG  — offline preview via trimesh's built-in renderer

Usage:
    from sakana_env import mesh3d
    fish = mesh3d.make_fish(color=(255, 122, 58))       # clownfish orange
    fish.export("assets/3d/fish_default.glb")

    leader = mesh3d.make_fish(color=(220, 53, 40))      # red leader
    leader.export("assets/3d/fish_leader.glb")

    shark = mesh3d.make_predator()
    shark.export("assets/3d/predator.glb")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


# ----------------------------------------------------------------------------
# Color palette (clownfish style, cel-shaded)
# ----------------------------------------------------------------------------

_ORANGE = np.array([255, 122, 58, 255], dtype=np.uint8)
_RED = np.array([220, 53, 40, 255], dtype=np.uint8)
_WHITE = np.array([250, 250, 250, 255], dtype=np.uint8)
_BLACK = np.array([20, 20, 20, 255], dtype=np.uint8)
_EYE_WHITE = np.array([248, 248, 250, 255], dtype=np.uint8)
_SHARK_GRAY = np.array([60, 65, 75, 255], dtype=np.uint8)


def _colored(mesh: trimesh.Trimesh, color: np.ndarray) -> trimesh.Trimesh:
    mesh.visual.face_colors = color
    return mesh


# ----------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------


def _ellipsoid(rx: float, ry: float, rz: float, subdivisions: int = 3) -> trimesh.Trimesh:
    """Uniform icosphere stretched to an ellipsoid."""
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    m.apply_scale([rx, ry, rz])
    return m


def _wedge(length: float, width: float, height: float,
           translate: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> trimesh.Trimesh:
    """Thin triangular wedge — used for fins and tail."""
    m = trimesh.creation.box(extents=[length, height, width])
    m.apply_translation(translate)
    return m


# ----------------------------------------------------------------------------
# Full fish
# ----------------------------------------------------------------------------


def _teardrop_body(length: float, height: float, width: float,
                   subdivisions: int = 3) -> trimesh.Trimesh:
    """Fish body: ellipsoid with asymmetric head/tail taper.

    Start from a unit icosphere, scale to length × height × width, then
    apply a per-vertex squash along +x (tail end) so the back half tapers
    smoothly into the tail attachment.

    Fish anatomy: wide at the head/middle, narrows toward the tail. Laterally
    compressed (width << height for most bony fish).
    """
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    m.apply_scale([length, height, width])
    # taper the tail half: scale y and z down as x becomes more negative
    v = m.vertices.copy()
    # normalize x over the length
    x_norm = v[:, 0] / length
    # taper factor: 1.0 at head (+x), 0.35 at tail (-x)
    taper = np.where(x_norm < 0,
                     1.0 + x_norm * 0.65,  # from 1.0 at x=0 to 0.35 at x=-length
                     1.0)
    v[:, 1] *= taper
    v[:, 2] *= taper
    m.vertices = v
    # shift so the tail end is at x=-length (head at x=+length*something)
    return m


def make_fish(
    color: tuple[int, int, int] = (255, 122, 58),
    with_stripes: bool = True,
    length: float = 3.0,
) -> trimesh.Trimesh:
    """Build a chibi 3D clownfish, length along +x.

    Correct fish proportions: **laterally compressed** (thin side-to-side),
    **tall** (body height ~0.75 of length), **tapered tail** (body narrows
    smoothly into the tail fin). Viewed from the side, you see the full
    profile; from above you see a thin sliver.

    Parts:
        body         — tapered teardrop ellipsoid, laterally compressed
        dorsal fin   — curved sail on top
        anal fin     — smaller fin on bottom
        pectoral fin — triangular fin on each side
        caudal fin   — fork tail (2 triangular plates)
        eye          — white sphere + black pupil, on both sides
        (optional) stripes — three curved white bands across the body
    """
    body_color = np.array([*color, 255], dtype=np.uint8)
    s = length / 3.0  # scale factor

    parts = []

    # -- BODY: tall, wide from the side, thin from top --
    # proportions: length 1.4, height 0.85, width 0.40
    body = _teardrop_body(1.4 * s, 0.85 * s, 0.40 * s, subdivisions=4)
    _colored(body, body_color)
    parts.append(body)

    # -- stripes: thin white slabs along the body, oriented vertically --
    if with_stripes:
        for cx in [0.55, 0.00, -0.50]:
            # height-aligned stripe, slightly scaled down in z so it sits on body
            stripe = trimesh.creation.box(
                extents=[0.12 * s, 1.8 * s, 0.46 * s]
            )
            stripe.apply_translation([cx * s, 0.0, 0.0])
            _colored(stripe, _WHITE)
            parts.append(stripe)

    # -- DORSAL (top) FIN: curved plate on top of body --
    dorsal_verts = np.array([
        [0.35, 0.78, 0.0], [-0.05, 1.10, 0.0], [-0.45, 0.95, 0.0],
        [-0.55, 0.78, 0.0], [0.0, 0.78, 0.0],
    ]) * s
    dorsal = _triangle_strip_fin(dorsal_verts, thickness=0.05 * s)
    _colored(dorsal, body_color)
    parts.append(dorsal)

    # -- ANAL (bottom) FIN: smaller bottom fin --
    anal_verts = np.array([
        [0.20, -0.80, 0.0], [-0.10, -1.05, 0.0], [-0.35, -0.95, 0.0],
        [-0.45, -0.80, 0.0],
    ]) * s
    anal = _triangle_strip_fin(anal_verts, thickness=0.05 * s)
    _colored(anal, body_color)
    parts.append(anal)

    # -- PECTORAL FINS: one on each side, angled outward --
    for side in [+1, -1]:
        pec_verts = np.array([
            [0.35, -0.05, side * 0.38],
            [0.0, -0.25, side * 0.65],
            [-0.10, 0.05, side * 0.55],
            [0.15, 0.10, side * 0.42],
        ]) * s
        pec = _triangle_strip_fin(pec_verts, thickness=0.04 * s, axis=np.array([0, 0, side]))
        _colored(pec, body_color)
        parts.append(pec)

    # -- CAUDAL (tail) FIN: fork, two triangles --
    for ty in [+1, -1]:
        tail_verts = np.array([
            [-1.10, ty * 0.05, 0.0],
            [-1.70, ty * 0.55, 0.0],
            [-1.85, ty * 0.35, 0.0],
            [-1.25, ty * 0.0, 0.0],
        ]) * s
        tail = _triangle_strip_fin(tail_verts, thickness=0.06 * s)
        _colored(tail, body_color)
        parts.append(tail)

    # -- EYES: big anime eyes on both sides, near the head --
    for z_side in [+0.35, -0.35]:
        eye_white = trimesh.creation.icosphere(subdivisions=2, radius=0.20 * s)
        eye_white.apply_translation([0.85 * s, 0.20 * s, z_side * s])
        _colored(eye_white, _EYE_WHITE)
        parts.append(eye_white)

        pupil = trimesh.creation.icosphere(subdivisions=2, radius=0.11 * s)
        pupil.apply_translation([0.95 * s, 0.18 * s, z_side * 1.08 * s])
        _colored(pupil, _BLACK)
        parts.append(pupil)

    fish = trimesh.util.concatenate(parts)
    return fish


def _triangle_strip_fin(verts: np.ndarray, thickness: float = 0.05,
                        axis: np.ndarray | None = None) -> trimesh.Trimesh:
    """Build a thin fin mesh from an ordered polygon outline.

    Extrudes the 2D outline into 3D by duplicating vertices along the
    given axis (default: z-axis). Returns a closed trimesh.
    """
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    axis = axis.astype(np.float64)
    # unit-ify axis
    n = np.linalg.norm(axis)
    if n > 0:
        axis = axis / n

    # front / back sheets
    front = verts + (thickness / 2) * axis
    back = verts - (thickness / 2) * axis
    all_verts = np.vstack([front, back])
    K = len(verts)

    # triangulate the front polygon as a fan
    faces_front = [[0, i, i + 1] for i in range(1, K - 1)]
    # back polygon fan (reversed winding)
    faces_back = [[K, K + i + 1, K + i] for i in range(1, K - 1)]
    # side strips connecting front to back
    faces_side = []
    for i in range(K):
        j = (i + 1) % K
        faces_side.append([i, j, K + j])
        faces_side.append([i, K + j, K + i])

    faces = np.array(faces_front + faces_back + faces_side, dtype=np.int64)
    return trimesh.Trimesh(vertices=all_verts, faces=faces, process=True)


def make_predator(length: float = 5.0) -> trimesh.Trimesh:
    """Build a chibi 3D shark-like predator, length along +x."""
    scale = length / 5.0

    # -- body: long tapered ellipsoid -----------------------------------
    body = _ellipsoid(2.3 * scale, 0.75 * scale, 0.7 * scale, subdivisions=3)
    _colored(body, _SHARK_GRAY)
    parts = [body]

    # -- dorsal fin -----------------------------------------------------
    dorsal = trimesh.creation.box(
        extents=[0.6 * scale, 0.9 * scale, 0.08 * scale]
    )
    dorsal.apply_translation([0.0, 0.9 * scale, 0.0])
    _colored(dorsal, _SHARK_GRAY)
    parts.append(dorsal)

    # -- pectoral fins --------------------------------------------------
    for z in [0.7, -0.7]:
        pec = trimesh.creation.box(
            extents=[0.65 * scale, 0.1 * scale, 0.4 * scale]
        )
        pec.apply_translation([0.1 * scale, -0.4 * scale, z * scale])
        _colored(pec, _SHARK_GRAY)
        parts.append(pec)

    # -- tail fork ------------------------------------------------------
    for ty in [0.45, -0.45]:
        tail = trimesh.creation.box(
            extents=[0.9 * scale, 0.55 * scale, 0.08 * scale]
        )
        angle = np.deg2rad(22 if ty > 0 else -22)
        Rz = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        tail.apply_transform(Rz)
        tail.apply_translation([-2.7 * scale, ty * scale, 0.0])
        _colored(tail, _SHARK_GRAY)
        parts.append(tail)

    # -- menacing red eyes ---------------------------------------------
    for z in [0.65, -0.65]:
        eye = trimesh.creation.icosphere(
            subdivisions=2, radius=0.14 * scale
        )
        eye.apply_translation([1.7 * scale, 0.25 * scale, z * scale])
        _colored(eye, _RED)
        parts.append(eye)

        pupil = trimesh.creation.icosphere(
            subdivisions=2, radius=0.07 * scale
        )
        pupil.apply_translation([1.78 * scale, 0.25 * scale, z * 1.05 * scale])
        _colored(pupil, _BLACK)
        parts.append(pupil)

    return trimesh.util.concatenate(parts)


# ----------------------------------------------------------------------------
# Export helpers
# ----------------------------------------------------------------------------


def save_glb(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def save_preview(mesh: trimesh.Trimesh, path: str | Path,
                 resolution: tuple[int, int] = (512, 512)) -> Path:
    """Render a static preview PNG via trimesh's built-in renderer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = mesh.scene()
    try:
        png = scene.save_image(resolution=resolution, visible=True)
        if png:
            path.write_bytes(png)
    except Exception as e:
        print(f"  preview render failed ({e}); skipping PNG")
    return path


__all__ = ["make_fish", "make_predator", "save_glb", "save_preview"]
