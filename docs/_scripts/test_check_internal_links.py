# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from docs._scripts.check_internal_links import (
    Link,
    check_links,
    extract_links,
    internal_path,
    load_route_index,
    slugify,
)


def write_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    docs = tmp_path / "docs"
    fern = docs / "fern"
    versions = fern / "versions"
    page = docs / "set-up" / "opensandbox.mdx"
    versions.mkdir(parents=True)
    page.parent.mkdir(parents=True)
    nav = versions / "latest.yml"
    nav.write_text(
        """
navigation:
  - section: Documentation
    contents:
      - section: Kubernetes Deployment
        slug: kubernetes-deployment
        contents:
          - section: Setup
            path: ../../set-up/index.mdx
            contents:
              - page: OpenSandbox
                slug: open-sandbox
                path: ../../set-up/opensandbox.mdx
""",
        encoding="utf-8",
    )
    (docs / "set-up" / "index.mdx").write_text("# Setup\n", encoding="utf-8")
    config = fern / "docs.yml"
    config.write_text(
        """
redirects:
  - source: "/documentation/self-managed-deployment/:path*"
    destination: "/documentation/kubernetes-deployment/:path*"
  - source: "/documentation/kubernetes-deployment/setup/opensandbox"
    destination: "/documentation/kubernetes-deployment/setup/open-sandbox"
""",
        encoding="utf-8",
    )
    return nav, config, page


def test_slugify_matches_fern_style() -> None:
    assert slugify("OpenSandbox") == "open-sandbox"
    assert slugify("Evaluate Agents & Models") == "evaluate-agents-and-models"
    assert slugify("API Reference") == "api-reference"


def test_load_route_index_maps_source_paths_and_redirects(tmp_path: Path) -> None:
    nav, config, page = write_config(tmp_path)

    index = load_route_index(nav, config)

    assert index.files[page.resolve()] == "/documentation/kubernetes-deployment/setup/open-sandbox"
    assert "/documentation/kubernetes-deployment/setup" in index.routes
    assert (
        index.resolve_redirect("/documentation/self-managed-deployment/setup/opensandbox")
        == "/documentation/kubernetes-deployment/setup/open-sandbox"
    )


def test_extract_links_ignores_code_and_supports_markdown_jsx_and_bare_urls(tmp_path: Path) -> None:
    page = tmp_path / "page.mdx"
    page.write_text(
        """
[Markdown](/documentation/markdown)
<Card href="/documentation/card">
https://docs.nvidia.com/nemo-platform/documentation/bare
`[Inline](/documentation/ignored-inline)`
```md
[Fenced](/documentation/ignored-fenced)
```
""",
        encoding="utf-8",
    )

    assert [(link.line, link.target) for link in extract_links(page)] == [
        (2, "/documentation/markdown"),
        (3, "/documentation/card"),
        (4, "https://docs.nvidia.com/nemo-platform/documentation/bare"),
    ]


def test_internal_path_accepts_only_docs_routes() -> None:
    assert internal_path("/documentation/page#section") == ("/documentation/page", "section")
    assert internal_path("https://docs.nvidia.com/nemo-platform/latest/documentation/page") == (
        "/latest/documentation/page",
        "",
    )
    assert internal_path("https://example.com/documentation/page") is None
    assert internal_path("/studio") is None


def test_check_links_distinguishes_canonical_redirected_and_broken(tmp_path: Path) -> None:
    nav, config, page = write_config(tmp_path)
    page.write_text(
        """
[Canonical](/documentation/kubernetes-deployment/setup/open-sandbox)
[Legacy](/documentation/self-managed-deployment/setup/opensandbox#install)
[Broken](/documentation/missing)
""",
        encoding="utf-8",
    )
    index = load_route_index(nav, config)

    redirected, broken = check_links(index)

    assert redirected == [
        (
            Link(page.resolve(), 3, "/documentation/self-managed-deployment/setup/opensandbox#install"),
            "/documentation/kubernetes-deployment/setup/open-sandbox#install",
        )
    ]
    assert broken == [(Link(page.resolve(), 4, "/documentation/missing"), "no matching Fern route or redirect")]
