"""
Build the three.js web demo rollout JSON.

Runs a boids rollout and exports the state trajectory as JSON. Meshy demo
assets already live in web/rollouts/ and are preserved by default.

Run:
    uv run python scripts/make_web_demo.py            # build only
    uv run python scripts/make_web_demo.py --serve    # build + serve at :8000
    uv run python scripts/make_web_demo.py --stage-procedural-assets --force-assets
"""

from __future__ import annotations

import argparse
import http.server
import shutil
import socketserver
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parents[1]
WEB_ROLLOUTS = ROOT / "web" / "rollouts"
ASSETS_3D = ROOT / "assets" / "3d"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def copy_asset(src: Path, dst: Path, *, force: bool = False) -> bool:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not force:
        print(f"  kept {display_path(dst)} ({dst.stat().st_size // 1024} KB)")
        return False
    shutil.copy2(src, dst)
    print(f"  staged {display_path(dst)} ({dst.stat().st_size // 1024} KB)")
    return True


def build_demo(
    n_steps: int = 240,
    *,
    out_dir: Path = WEB_ROLLOUTS,
    stage_procedural_assets: bool = False,
    force_assets: bool = False,
) -> Path:
    """Run boids rollout and export JSON for the web viewer."""
    from sakana_env import boids, env, viz_web

    cfg = env.EnvConfig(
        n_fish=42,
        n_predators=2,
        n_food=18,
        world_size=80.0,
        max_steps=n_steps,
    )

    @jax.jit
    def step_fn(state, key):
        action = boids.boids_action(state, cfg)
        return env.step(state, action, key, cfg)

    key = jax.random.PRNGKey(11)
    state = env.reset(key, cfg)

    print(f"rollout: {cfg.n_fish} fish, {cfg.n_predators} preds, {n_steps} steps")
    t0 = time.time()
    states = [state]
    for _ in range(n_steps):
        key, sk = jax.random.split(key)
        state, reward, done = step_fn(state, sk)
        states.append(state)
    jax.block_until_ready(state.fish_pos)
    print(f"  rollout time: {time.time() - t0:.2f}s")
    print(f"  survivors: {int(state.fish_alive.sum())}/{cfg.n_fish}")

    # JSON export
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "boids.json"
    viz_web.export_trajectory(states, cfg, str(json_out))
    print(f"  wrote {json_out} ({json_out.stat().st_size // 1024} KB)")

    if stage_procedural_assets:
        for name in ["fish_default.glb", "fish_leader.glb", "predator.glb"]:
            copy_asset(ASSETS_3D / name, out_dir / name, force=force_assets)

    return json_out


def serve(port: int = 8000) -> None:
    """Serve web/ at localhost:port and open the browser."""
    web_dir = ROOT / "web"
    handler = http.server.SimpleHTTPRequestHandler

    class Handler(handler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_dir), **kwargs)

    url = f"http://localhost:{port}/"
    print(f"serving {web_dir} at {url}")
    print("  (Ctrl+C to stop)")
    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            webbrowser.open(url)
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="start local webserver after build")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--out-dir", type=Path, default=WEB_ROLLOUTS)
    parser.add_argument("--stage-procedural-assets", action="store_true", help="copy procedural GLBs into web/rollouts")
    parser.add_argument("--force-assets", action="store_true", help="overwrite existing staged GLBs")
    args = parser.parse_args()

    build_demo(
        n_steps=args.steps,
        out_dir=args.out_dir,
        stage_procedural_assets=args.stage_procedural_assets,
        force_assets=args.force_assets,
    )
    if args.serve:
        serve(port=args.port)


if __name__ == "__main__":
    main()
