#!/usr/bin/env python
"""Record the Three.js demo to MP4."""

from __future__ import annotations

import argparse
import http.server
import shutil
import socket
import subprocess
import tempfile
import threading
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(port: int) -> http.server.ThreadingHTTPServer:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def record_webm(url: str, out_dir: Path, seconds: float, width: int, height: int) -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("playwright is missing. Run: uv run --extra video python scripts/capture_web_video.py") from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(out_dir),
            record_video_size={"width": width, "height": height},
        )
        page = context.new_page()
        video = page.video
        try:
            page.goto(url, wait_until="load", timeout=30_000)
            page.wait_for_function("document.querySelectorAll('canvas').length > 0", timeout=30_000)
            page.wait_for_function("document.querySelector('#info')?.textContent?.includes('step')", timeout=60_000)
            page.wait_for_timeout(int(seconds * 1000))
            info = page.evaluate("document.querySelector('#info')?.textContent || document.body.innerText.slice(0, 80)")
            print(f"  captured through {info}")
        finally:
            context.close()
            browser.close()
        return Path(video.path())


def convert_to_mp4(webm: Path, mp4: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is missing")
    mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(webm),
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture SakanaEnv web demo video.")
    parser.add_argument("--out", default="artifacts/sakanaenv_demo.mp4")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    port = args.port or free_port()
    server = start_server(port)
    try:
        with tempfile.TemporaryDirectory(prefix="sakanaenv-video-") as tmp:
            print(f"serving {WEB} at http://127.0.0.1:{port}/")
            webm = record_webm(f"http://127.0.0.1:{port}/", Path(tmp), args.seconds, args.width, args.height)
            out = Path(args.out)
            convert_to_mp4(webm, out)
            print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
