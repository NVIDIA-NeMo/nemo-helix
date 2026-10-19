# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MDX to Markdown conversion used by `nemo docs`."""

from __future__ import annotations

from pathlib import Path

from nemo_helix_ext.cli.core.mdx_markdown import DOCS_SITE_URL, mdx_to_markdown

FRONTMATTER = """---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

title: "Page Title"
description: ""
---
"""


def _write_snippet(docs_root: Path, rel: str, content: str) -> None:
    path = docs_root / "fern" / "snippets" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestFrontmatterAndComments:
    def test_frontmatter_title_becomes_h1(self):
        result = mdx_to_markdown(FRONTMATTER + "\nBody text.\n")

        assert result == "# Page Title\n\nBody text.\n"

    def test_existing_h1_is_not_duplicated(self):
        result = mdx_to_markdown(FRONTMATTER + "# Own Heading\n\nBody.\n")

        assert result == "# Own Heading\n\nBody.\n"

    def test_empty_title_is_skipped(self):
        result = mdx_to_markdown('---\ntitle: ""\n---\nBody.\n')

        assert result == "Body.\n"

    def test_unquoted_title(self):
        assert mdx_to_markdown("---\ntitle: Plain\n---\nBody.\n").startswith("# Plain\n")

    def test_mdx_comments_removed(self):
        source = "{/* @nemo-docs: skip-python-type-check */}\nKeep {/* inline */}this.\n"

        assert mdx_to_markdown(source) == "Keep this.\n"

    def test_multiline_comment_removed(self):
        source = "Before.\n\n{/* first line\nsecond line */}\n\nAfter.\n"

        assert mdx_to_markdown(source) == "Before.\n\nAfter.\n"


class TestCodeIsPreserved:
    def test_fenced_code_untouched(self):
        source = "```bash\n<Note>\n{/* keep */}\nexport HF_TOKEN=<YOUR_TOKEN>\n---\n```\n"

        assert mdx_to_markdown(source) == source

    def test_tilde_fence_and_longer_fence(self):
        source = "````md\n```\n<Tip>\n```\n````\n~~~\n<Warning>\n~~~\n"

        assert mdx_to_markdown(source) == source

    def test_inline_code_untouched(self):
        source = "Press `<Tab>` and use `{/* x */}` or `<span>`.\n"

        assert mdx_to_markdown(source) == source


class TestCallouts:
    def test_note_becomes_labelled_blockquote(self):
        source = "<Note>\nRemember this.\n</Note>\n\nAfter.\n"

        assert mdx_to_markdown(source) == "> **Note:**\n> Remember this.\n\nAfter.\n"

    def test_titled_warning_with_code_block(self):
        source = '<Warning title="Careful">\n\nText.\n\n```bash\nrm -rf x\n```\n\n</Warning>\n'

        assert mdx_to_markdown(source) == ("> **Warning: Careful**\n>\n> Text.\n>\n> ```bash\n> rm -rf x\n> ```\n")

    def test_tip_inside_tab(self):
        source = '<Tabs>\n<Tab title="One">\n<Tip>\nHint.\n</Tip>\n</Tab>\n</Tabs>\n'

        assert mdx_to_markdown(source) == "**One**\n\n> **Tip:**\n> Hint.\n"


class TestComponents:
    def test_tabs_become_bold_titles(self):
        source = (
            '<Tabs>\n\n<Tab title="uv">\n\nUse uv.\n\n</Tab>\n\n<Tab title="pip">\n\nUse pip.\n\n</Tab>\n\n</Tabs>\n'
        )

        assert mdx_to_markdown(source) == "**uv**\n\nUse uv.\n\n**pip**\n\nUse pip.\n"

    def test_accordion_becomes_bold_title(self):
        source = '<Accordion title="Details">\n\nHidden text.\n\n</Accordion>\n'

        assert mdx_to_markdown(source) == "**Details**\n\nHidden text.\n"

    def test_card_with_body(self):
        source = (
            '<Cards>\n\n<Card title="Quickstart" href="/documentation/start">\n\n'
            "Get started.\n\n"
            '<small><span class="md-tag">standalone</span> <span class="md-tag">beginner</span></small>\n\n'
            "</Card>\n\n</Cards>\n"
        )

        assert mdx_to_markdown(source) == (
            f"**[Quickstart]({DOCS_SITE_URL}/documentation/start)**\n\nGet started.\n\n`standalone` `beginner`\n"
        )

    def test_self_closing_cards_become_list(self):
        source = '<Cards>\n  <Card title="A" href="https://a.example" />\n  <Card title="B" href="/documentation/b" />\n</Cards>\n'

        assert mdx_to_markdown(source) == f"- [A](https://a.example)\n- [B]({DOCS_SITE_URL}/documentation/b)\n"

    def test_button_becomes_link(self):
        source = '<Button href="/documentation/x" variant="primary">**Quickstart**</Button>\n'

        assert mdx_to_markdown(source) == f"[**Quickstart**]({DOCS_SITE_URL}/documentation/x)\n"

    def test_steps_are_numbered(self):
        source = (
            '<Steps>\n<Step title="Install">\nRun it.\n</Step>\n<Step title="Verify">\nCheck it.\n</Step>\n</Steps>\n'
        )

        assert mdx_to_markdown(source) == "**Step 1: Install**\nRun it.\n**Step 2: Verify**\nCheck it.\n"

    def test_unknown_component_degrades_to_inner_text(self):
        source = '<Frame caption="x">\n\nInner **text**.\n\n</Frame>\n\nInline <Badge color="green">new</Badge> item.\n'

        assert mdx_to_markdown(source) == "Inner **text**.\n\nInline new item.\n"


class TestHtml:
    def test_anchor_ids_dropped(self):
        source = '<a id="section"></a>\n## Section\n'

        assert mdx_to_markdown(source) == "## Section\n"

    def test_inline_anchor_becomes_link(self):
        source = 'See <a href="https://example.com" target="_blank">the site</a>.\n'

        assert mdx_to_markdown(source) == "See [the site](https://example.com).\n"

    def test_multiline_anchor_in_div(self):
        source = (
            '<div className="notebook-actions">\n'
            '  <a href="https://colab.example/nb.ipynb" target="_blank" className="btn">\n'
            '    <span aria-hidden="true">&#9654;</span><span>Run in Google Colab</span>\n'
            "  </a>\n"
            "</div>\n\nNext.\n"
        )

        assert mdx_to_markdown(source) == "[Run in Google Colab](https://colab.example/nb.ipynb)\n\nNext.\n"

    def test_multiline_anchor_with_inline_code(self):
        source = '<a href="/documentation/x">\n  Run `nemo setup`\n</a>\n'

        assert mdx_to_markdown(source) == f"[Run `nemo setup`]({DOCS_SITE_URL}/documentation/x)\n"

    def test_multiline_anchor_keeps_each_lines_inline_code(self):
        source = '<a href="https://x.example">\n  Run `a`\n</a> then `b`\n'

        assert mdx_to_markdown(source) == "[Run `a`](https://x.example) then `b`\n"


class TestLinks:
    def test_site_relative_links_made_absolute(self):
        source = "Read [Setup](/documentation/get-started/setup) and [ext](https://x.example) and [img](../img.png).\n"

        assert mdx_to_markdown(source) == (
            f"Read [Setup]({DOCS_SITE_URL}/documentation/get-started/setup) and [ext](https://x.example) "
            "and [img](../img.png).\n"
        )


class TestSnippets:
    def test_snippet_expanded_without_its_frontmatter(self, tmp_path: Path):
        _write_snippet(
            tmp_path, "_snippets/note.mdx", '---\ntitle: ""\n---\n<Note>\nUse the default workspace.\n</Note>\n'
        )
        source = 'Intro.\n\n<Markdown src="/snippets/_snippets/note.mdx" />\n\nAfter.\n'

        assert mdx_to_markdown(source, docs_root=tmp_path) == (
            "Intro.\n\n> **Note:**\n> Use the default workspace.\n\nAfter.\n"
        )

    def test_snippet_inside_callout_is_quoted(self, tmp_path: Path):
        _write_snippet(tmp_path, "s.mdx", "Shared line.\n")
        source = '<Tip>\n<Markdown src="/snippets/s.mdx" />\n</Tip>\n'

        assert mdx_to_markdown(source, docs_root=tmp_path) == "> **Tip:**\n> Shared line.\n"

    def test_nested_snippets_expand(self, tmp_path: Path):
        _write_snippet(tmp_path, "outer.mdx", 'Outer.\n<Markdown src="/snippets/inner.mdx" />\n')
        _write_snippet(tmp_path, "inner.mdx", "Inner.\n")

        result = mdx_to_markdown('<Markdown src="/snippets/outer.mdx" />\n', docs_root=tmp_path)

        assert result == "Outer.\nInner.\n"

    def test_recursive_snippet_terminates(self, tmp_path: Path):
        _write_snippet(tmp_path, "loop.mdx", 'Loop.\n<Markdown src="/snippets/loop.mdx" />\n')

        result = mdx_to_markdown('<Markdown src="/snippets/loop.mdx" />\n', docs_root=tmp_path)

        assert result.count("Loop.") == 5

    def test_missing_or_escaping_snippet_is_dropped(self, tmp_path: Path):
        (tmp_path / "secret.mdx").write_text("Secret.\n")
        source = 'A.\n<Markdown src="/snippets/missing.mdx" />\n<Markdown src="/snippets/../../secret.mdx" />\nB.\n'

        assert mdx_to_markdown(source, docs_root=tmp_path) == "A.\nB.\n"

    def test_snippet_without_docs_root_is_dropped(self):
        assert mdx_to_markdown('A.\n<Markdown src="/snippets/x.mdx" />\n') == "A.\n"
