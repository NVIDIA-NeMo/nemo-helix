# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check visible language tabs on a Fern preview or published API page.

Run once per affected page, using the matching version's OpenAPI spec::

    uv run --with playwright python docs/_scripts/check_rendered_api_samples.py \
        --url https://PREVIEW/documentation/reference/api-reference/files/upload-file \
        --path '/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}' --method put

Install Chromium first with ``uv run --with playwright playwright install chromium``.
This checks rendered, visible code after switching languages, not hidden serialized
page data (which can contain correct custom samples while the UI shows fallbacks).
It never executes the API snippets or sends requests to the documented deployment.
"""

import argparse
import re
from pathlib import Path

import yaml


def normalized(source: str) -> str:
    return re.sub(r"\s+", " ", source).strip()


def check_page(page, url: str, samples: dict[str, str]) -> None:
    consent = page.get_by_role("button", name="Reject Optional", exact=True)
    page.add_locator_handler(consent, lambda: consent.click())
    page.goto(url, wait_until="load")
    languages = re.compile(r"^(Python|JavaScript|cURL)$")
    page.get_by_role("tab", name=languages).or_(page.get_by_role("button", name=languages)).first.wait_for()
    for language, expected in samples.items():
        label = {"python": "Python", "javascript": "JavaScript", "curl": "cURL"}[language]
        # Fern versions expose either tabs or a language selector menu.
        tab = page.get_by_role("tab", name=label, exact=True)
        if tab.count():
            tab.first.click()
        else:
            selector = page.get_by_role("button", name=re.compile(r"^(Python|JavaScript|cURL)$"))
            selector.first.click()
            page.get_by_text(label, exact=True).last.click()
        try:
            page.wait_for_function(
                """expected => [...document.querySelectorAll('pre')].some(pre =>
                    pre.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}) &&
                    pre.innerText.replace(/\\s+/g, ' ').trim().includes(expected))""",
                arg=normalized(expected),
                timeout=15000,
            )
        except Exception as exc:
            raise AssertionError(f"{url}: visible {label} sample does not match the spec override") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Full preview/published API page URL")
    parser.add_argument("--path", required=True, help="OpenAPI path template")
    parser.add_argument("--method", required=True)
    parser.add_argument("--spec", type=Path, default=Path("openapi/openapi.yaml"))
    args = parser.parse_args()
    operation = yaml.safe_load(args.spec.read_text())["paths"][args.path][args.method.lower()]
    samples = {
        sample["lang"].lower(): sample["source"]
        for sample in operation["x-codeSamples"]
        if sample["lang"] in {"Python", "JavaScript"}
    }
    assert set(samples) == {"python", "javascript"}, "Expected both custom language samples"
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            check_page(browser.new_page(), args.url, samples)
        finally:
            browser.close()
    print(f"Verified visible Python and JavaScript samples: {args.url}")


if __name__ == "__main__":
    main()
