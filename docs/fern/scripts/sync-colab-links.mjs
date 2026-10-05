#!/usr/bin/env node
/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Keep Fern MDX "Run in Google Colab" affordances derived from adjacent
 * notebooks instead of hand-maintained per page.
 *
 * - Inline MDX pages with an adjacent .ipynb get a top-of-page markdown link.
 * - NotebookViewer wrapper pages with an adjacent .ipynb get a colabUrl prop.
 * - Pages without an adjacent .ipynb do not get a generated affordance.
 */

import { execFileSync } from "node:child_process";
import { readdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(SCRIPT_DIR, "../../..");
const DOCS_ROOT = join(REPO_ROOT, "docs");
const COLAB_PREFIX = "https://colab.research.google.com/github/NVIDIA-NeMo/nemo-helix/blob";
const argv = process.argv.slice(2);
const CHECK = argv.includes("--check");
const docsRootArg = valueForFlag("--docs-root");
const colabRefArg = valueForFlag("--ref");
const TARGET_DOCS_ROOT = docsRootArg ? resolve(docsRootArg) : DOCS_ROOT;
const COLAB_REF = colabRefArg ?? detectColabRef();

const TOP_COLAB_LINK_RE = /^\n*\[Run in Google Colab\]\(https:\/\/colab\.research\.google\.com\/github\/NVIDIA-NeMo\/nemo-helix\/blob\/[^\s)]+\)\n{1,2}/;
const NOTEBOOK_VIEWER_RE = /<NotebookViewer\b[\s\S]*?\/>/g;
const COLAB_URL_PROP_RE = /\n\s*colabUrl="[^"]*"/;

function valueForFlag(flag) {
  const index = argv.indexOf(flag);
  if (index === -1) {
    return undefined;
  }
  const value = argv[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
}

function detectColabRef() {
  if (process.env.NHX_DOCS_COLAB_REF) {
    return process.env.NHX_DOCS_COLAB_REF;
  }
  if (process.env.GITHUB_REF_TYPE === "tag" && process.env.GITHUB_REF_NAME) {
    return process.env.GITHUB_REF_NAME;
  }
  if (process.env.GITHUB_REF_NAME?.startsWith("release/")) {
    return process.env.GITHUB_REF_NAME;
  }

  try {
    const branch = execFileSync("git", ["branch", "--show-current"], {
      cwd: REPO_ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    if (branch?.startsWith("release/")) {
      return branch;
    }
  } catch {
    // Fall through to the public default branch.
  }

  return "main";
}

function slashRelative(from, to) {
  return relative(from, to).split(sep).join("/");
}

function repoRelative(path) {
  return slashRelative(REPO_ROOT, path);
}

function notebookSourcePath(notebookPath) {
  if (TARGET_DOCS_ROOT === DOCS_ROOT) {
    return repoRelative(notebookPath);
  }
  return `docs/${slashRelative(TARGET_DOCS_ROOT, notebookPath)}`;
}

function colabUrlFor(notebookPath) {
  return `${COLAB_PREFIX}/${COLAB_REF}/${notebookSourcePath(notebookPath)}`;
}

function splitFrontmatter(source) {
  if (!source.startsWith("---\n")) {
    return { frontmatter: "", body: source };
  }
  const end = source.indexOf("\n---", 4);
  if (end === -1) {
    return { frontmatter: "", body: source };
  }
  const closeEnd = source.indexOf("\n", end + 4);
  if (closeEnd === -1) {
    return { frontmatter: source, body: "" };
  }
  return {
    frontmatter: source.slice(0, closeEnd + 1),
    body: source.slice(closeEnd + 1),
  };
}

function syncMarkdownColabLink(source, notebookPath) {
  const { frontmatter, body } = splitFrontmatter(source);
  const bodyWithoutTopLink = body.replace(TOP_COLAB_LINK_RE, "");
  if (!notebookPath) {
    return `${frontmatter}${bodyWithoutTopLink}`;
  }
  const link = `[Run in Google Colab](${colabUrlFor(notebookPath)})`;
  return `${frontmatter}\n${link}\n\n${bodyWithoutTopLink.replace(/^\n+/, "")}`;
}

function syncNotebookViewerColabUrl(source, notebookPath) {
  return source.replace(NOTEBOOK_VIEWER_RE, (viewer) => {
    const withoutColab = viewer.replace(COLAB_URL_PROP_RE, "");
    if (!notebookPath) {
      return withoutColab;
    }
    const indentMatch = withoutColab.match(/\n(\s*)name=/);
    const indent = indentMatch?.[1] ?? "  ";
    return withoutColab.replace(
      /\s*\/>$/,
      `\n${indent}colabUrl="${colabUrlFor(notebookPath)}"\n/>`,
    );
  });
}

async function pathExists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function* walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (TARGET_DOCS_ROOT === DOCS_ROOT && path === join(DOCS_ROOT, "fern")) {
        continue;
      }
      yield* walk(path);
    } else if (entry.isFile() && entry.name.endsWith(".mdx")) {
      yield path;
    }
  }
}

async function main() {
  const changed = [];

  for await (const mdxPath of walk(TARGET_DOCS_ROOT)) {
    const source = await readFile(mdxPath, "utf8");
    const adjacentNotebook = mdxPath.replace(/\.mdx$/, ".ipynb");
    const notebookPath = (await pathExists(adjacentNotebook)) ? adjacentNotebook : null;
    const hasNotebookViewer = NOTEBOOK_VIEWER_RE.test(source);
    NOTEBOOK_VIEWER_RE.lastIndex = 0;

    const updated = hasNotebookViewer
      ? syncNotebookViewerColabUrl(source, notebookPath)
      : syncMarkdownColabLink(source, notebookPath);

    if (updated !== source) {
      changed.push(repoRelative(mdxPath));
      if (!CHECK) {
        await writeFile(mdxPath, updated);
      }
    }
  }

  if (changed.length === 0) {
    console.log("sync-colab-links: all MDX Colab links are up to date");
    return;
  }

  for (const path of changed) {
    console.log(`${CHECK ? "stale" : "updated"} ${path}`);
  }

  if (CHECK) {
    console.error(
      `\nsync-colab-links: ${changed.length} file(s) need generated Colab link updates. ` +
        "Run: npm --prefix docs/fern run sync:colab-links",
    );
    process.exit(1);
  }

  console.log(`sync-colab-links: updated ${changed.length} file(s)`);
}

await main();
