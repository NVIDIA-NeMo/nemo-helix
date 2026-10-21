# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runs in a separate Playwright environment, not in the Helix tool environment."""

import os
from pathlib import Path

# Installed only in the isolated browser environment, not the source workspace.
from playwright.sync_api import expect, sync_playwright  # ty: ignore[unresolved-import]

with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        response = page.goto(f"{os.environ['NHX_BASE_URL']}/studio/workspaces/release-smoke/dashboard", timeout=60000)
        assert response and response.ok, "Studio document did not load"
        # Fresh browsers show onboarding over the rendered dashboard.
        expect(page.get_by_role("heading", name="Quickstart", exact=True, include_hidden=True)).to_be_visible(
            timeout=60000
        )
        expect(page.get_by_role("link", name="Agents", exact=True, include_hidden=True)).to_be_visible()
        assert not errors, f"Studio JavaScript errors: {errors}"
    finally:
        page.screenshot(path=str(Path(os.environ["SMOKE_RESULTS"], "studio.png")), full_page=True)
        browser.close()
