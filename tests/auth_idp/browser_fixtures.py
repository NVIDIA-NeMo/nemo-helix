# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import threading
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from playwright.sync_api import BrowserContext, Error, Page, sync_playwright

BROWSER_ARTIFACTS_ENV = "NHX_AUTH_IDP_BROWSER_ARTIFACTS_DIR"


@dataclass(frozen=True)
class AuthBrowser:
    context: BrowserContext
    page: Page


class _CrossOriginHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b"<!doctype html><title>Cross-origin auth test</title>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def auth_browser(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[AuthBrowser]:
    configured_artifacts = os.environ.get(BROWSER_ARTIFACTS_ENV)
    artifacts_dir = Path(configured_artifacts) if configured_artifacts else tmp_path / "browser"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.node.nodeid).strip("-")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(ignore_https_errors=True)
        context.add_init_script("try { localStorage.setItem('tour-seen', 'true'); } catch { /* opaque origin */ }")
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        console_messages: list[str] = []
        page.on("console", lambda message: console_messages.append(f"{message.type}: {message.text}"))
        try:
            yield AuthBrowser(context=context, page=page)
        finally:
            with suppress(Error):
                page.screenshot(path=artifacts_dir / f"{artifact_name}.png", full_page=True)
            with suppress(Error):
                context.tracing.stop(path=artifacts_dir / f"{artifact_name}.zip")
            (artifacts_dir / f"{artifact_name}.console.log").write_text(
                "\n".join(console_messages),
                encoding="utf-8",
            )
            with suppress(Error):
                context.close()
            with suppress(Error):
                browser.close()


@pytest.fixture
def cross_origin_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CrossOriginHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
