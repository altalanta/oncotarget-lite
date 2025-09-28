"""Utility to start the Streamlit app and capture a screenshot via Playwright."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

from playwright.async_api import async_playwright

from .utils import ensure_dir

APP_PATH = Path(__file__).parent / "streamlit_app.py"
STREAMLIT_CMD = [
    "streamlit",
    "run",
    str(APP_PATH),
    "--server.headless",
    "true",
    "--server.port",
    "8501",
]


class SnapshotError(RuntimeError):
    """Raised when the Streamlit snapshot could not be captured."""


def _wait_for_server(url: str, timeout: int) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with contextlib.closing(urlopen(url)):
                return
        except URLError:
            time.sleep(0.5)
    raise SnapshotError("Streamlit server did not start within timeout")


async def _capture(url: str, output_path: Path) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        await page.set_viewport_size({"width": 1280, "height": 720})
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(output_path), full_page=True)
        await browser.close()


def capture_streamlit(*, output_path: Path, timeout: int = 30) -> Path:
    """Launch the Streamlit app in a subprocess and capture a screenshot."""

    ensure_dir(output_path.parent)
    with subprocess.Popen(
        STREAMLIT_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        try:
            _wait_for_server("http://localhost:8501/healthz", timeout)
            asyncio.run(_capture("http://localhost:8501", output_path))
            return output_path
        finally:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)

