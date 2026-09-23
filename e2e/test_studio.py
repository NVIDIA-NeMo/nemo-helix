# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the Studio UI service.

These tests verify that the Studio static files are correctly served
and that the production build is properly configured.

Note: These tests require the Studio UI to be built and available.
All tests self-skip when Studio static files are not mounted.
"""

import re

from nemo_helix import NeMoHelix
from nhx.testing.pytest_outcomes import pytest_skip


def _studio_available(sdk: NeMoHelix) -> bool:
    """Check if the Studio UI is available (static files are mounted)."""
    response = sdk._client.get("/studio/")
    return response.status_code == 200


def test_studio_index_html(sdk: NeMoHelix):
    """Test that /studio/ serves the index.html correctly.

    This verifies:
    1. The Studio service is mounted and serving static files from static_files_path
    2. The response is HTML with correct content-type
    3. Assets are prefixed with /studio/ (correct Vite base URL)
    4. No unreplaced STUDIO_UI_ markers remain in the HTML
    """
    if not _studio_available(sdk):
        pytest_skip("Studio UI not available (static files not mounted)")

    response = sdk._client.get("/studio/")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert "text/html" in response.headers.get("content-type", ""), (
        f"Expected text/html content-type, got: {response.headers.get('content-type')}"
    )

    html = response.text

    # Basic HTML structure checks
    assert "<html" in html.lower(), "Response doesn't appear to be HTML"
    assert "<script" in html.lower(), "HTML doesn't contain script tags"

    # Assets should be prefixed with /studio/
    assert "/studio/assets/" in html, (
        "Assets are not prefixed with /studio/. Vite wasn't built with --mode fastapi or VITE_BASE_URL wasn't set."
    )

    # No unreplaced STUDIO_UI_ markers
    assert "STUDIO_UI_" not in html, "Found unreplaced STUDIO_UI_ marker in index.html"


def test_studio_spa_routing(sdk: NeMoHelix):
    """Test that SPA routing returns index.html for client-side routes.

    This verifies that the static_files_path is correctly configured and
    the SPA fallback mechanism works - non-file paths should return index.html
    to allow client-side routing to handle them.
    """
    if not _studio_available(sdk):
        pytest_skip("Studio UI not available (static files not mounted)")

    # These paths don't exist as files but should return index.html for SPA routing
    spa_routes = [
        "/studio/dashboard",
        "/studio/workspaces/test-workspace",
        "/studio/workspaces/test/models",
    ]

    for route in spa_routes:
        response = sdk._client.get(route)
        assert response.status_code == 200, f"SPA route {route} returned {response.status_code}"
        assert "text/html" in response.headers.get("content-type", ""), (
            f"SPA route {route} didn't return HTML content-type"
        )
        # Verify it's actually the index.html (contains script tags for the app)
        assert "<script" in response.text.lower(), f"SPA route {route} didn't return app HTML"


def test_studio_js_bundle(sdk: NeMoHelix):
    """Test that the Studio JS bundle is production-ready.

    This verifies:
    1. JS assets load successfully from /studio/assets/
    2. No unreplaced STUDIO_UI_ markers in the bundle
    3. VITE_APP_ENV=production is baked into the build
    """
    if not _studio_available(sdk):
        pytest_skip("Studio UI not available (static files not mounted)")

    # Get index.html to find all JS asset paths
    index_response = sdk._client.get("/studio/")
    assert index_response.status_code == 200

    # Find all JS asset paths (e.g., src="/studio/assets/index-XXXXX.js")
    js_paths = re.findall(r'(?:src|href)="(/studio/assets/[^"]+\.js)"', index_response.text)
    assert js_paths, "No JS assets found in index.html"

    all_js_content = ""
    for js_path in js_paths:
        js_response = sdk._client.get(js_path)
        assert js_response.status_code == 200, f"Failed to load JS asset at {js_path}"
        all_js_content += js_response.text

    # No unreplaced STUDIO_UI_ markers
    assert "STUDIO_UI_" not in all_js_content, "Found unreplaced STUDIO_UI_ marker in JS bundle"

    # Production environment baked in (may be in any chunk, not just the entry)
    assert "production" in all_js_content.lower(), (
        "JS bundle doesn't contain 'production' environment. Ensure VITE_APP_ENV=production is set in .env.fastapi"
    )


def test_studio_css_assets(sdk: NeMoHelix):
    """Test that CSS assets are served correctly from static_files_path.

    This verifies that the static file serving works for different asset types.
    """
    if not _studio_available(sdk):
        pytest_skip("Studio UI not available (static files not mounted)")

    # Get index.html to find CSS asset path
    index_response = sdk._client.get("/studio/")
    assert index_response.status_code == 200

    # Find a CSS asset path (e.g., href="/studio/assets/index-XXXXX.css")
    css_match = re.search(r'href="(/studio/assets/[^"]+\.css)"', index_response.text)
    if css_match:
        css_path = css_match.group(1)
        css_response = sdk._client.get(css_path)
        assert css_response.status_code == 200, f"Failed to load CSS asset at {css_path}"
        assert "text/css" in css_response.headers.get("content-type", ""), "CSS asset didn't have correct content-type"
