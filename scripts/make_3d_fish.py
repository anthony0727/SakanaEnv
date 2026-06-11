"""
Generate procedural 3D fish + predator meshes, export as GLB for three.js.

This replaces (or rather, stands alongside) the 2D procedural sprites in
scripts/make_anime_sprite.py. Run once to produce the 3D assets the web
renderer consumes.

Output:
    assets/3d/fish_default.glb   — orange chibi clownfish
    assets/3d/fish_leader.glb    — red leader variant
    assets/3d/predator.glb       — dark shark-like silhouette
    assets/3d/*.png              — preview renders (if pyglet available)
"""

from __future__ import annotations

from pathlib import Path

from sakana_env import mesh3d


OUT = Path(__file__).resolve().parents[1] / "assets" / "3d"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("generating 3D meshes...")

    default = mesh3d.make_fish(color=(255, 122, 58))
    mesh3d.save_glb(default, OUT / "fish_default.glb")
    print(f"  fish_default.glb   vertices={len(default.vertices)}  faces={len(default.faces)}")

    leader = mesh3d.make_fish(color=(220, 53, 40))
    mesh3d.save_glb(leader, OUT / "fish_leader.glb")
    print(f"  fish_leader.glb    vertices={len(leader.vertices)}  faces={len(leader.faces)}")

    predator = mesh3d.make_predator()
    mesh3d.save_glb(predator, OUT / "predator.glb")
    print(f"  predator.glb       vertices={len(predator.vertices)}  faces={len(predator.faces)}")

    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
