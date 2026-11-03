# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Browser-fixture tests; run with Playwright and Chromium installed."""

import pytest

from docs._scripts.check_rendered_api_samples import check_page


@pytest.fixture
def page():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            yield browser.new_page()
        finally:
            browser.close()


@pytest.mark.parametrize("dropdown", [False, True], ids=["tabs", "language-menu"])
def test_visible_samples_after_language_switch(page, dropdown):
    if dropdown:
        controls = """<button id="selector" onclick="menu.hidden=false">cURL</button>
<div id="menu" hidden>
<button onclick="selectLanguage('Python')">Python</button>
<button onclick="selectLanguage('JavaScript')">JavaScript</button>
</div>"""
    else:
        controls = """<button role="tab" onclick="selectLanguage('Python')">Python</button>
<button role="tab" onclick="selectLanguage('JavaScript')">JavaScript</button>"""
    html = (
        controls
        + """<pre id="sample">curl</pre>
<script>
function selectLanguage(language) {
  sample.innerText = language === 'Python' ? "print(response.status_code)" : "console.log(response.status);";
  if (typeof menu !== 'undefined') { menu.hidden = true; selector.innerText = language; }
}
</script>"""
    )
    page.route("https://preview.test/**", lambda route: route.fulfill(body=html, content_type="text/html"))
    check_page(
        page,
        "https://preview.test/upload",
        {"python": "print(response.status_code)", "javascript": "console.log(response.status);"},
    )


@pytest.mark.parametrize("style", ["display:none", "visibility:hidden", "opacity:0"])
def test_hidden_correct_sample_cannot_mask_visible_fallback(page, style):
    html = f"""<button role="tab">Python</button>
<pre style="{style}">print(response.status_code)</pre>
<pre>print(response.json())</pre>"""
    page.route("https://preview.test/**", lambda route: route.fulfill(body=html, content_type="text/html"))
    with pytest.raises(AssertionError, match="visible Python sample does not match"):
        check_page(page, "https://preview.test/delete", {"python": "print(response.status_code)"})
