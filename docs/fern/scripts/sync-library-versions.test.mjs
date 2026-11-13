// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { buildLibraryVersionsMdx, libraryVersionSourcePaths, syncLibraryVersions } from "./sync-library-versions.mjs";

function pkg(name, version, source = '{ registry = "https://pypi.org/simple" }') {
  return `[[package]]\nname = "${name}"\nversion = "${version}"\nsource = ${source}\n`;
}

function render(lock, others = []) {
  return buildLibraryVersionsMdx(new Map([["uv.lock", lock], ...others]));
}

test("discovers library families and new NeMo packages without a version allowlist", () => {
  const names = ["nemo-anonymizer", "data-designer", "data-designer-config", "data-designer-engine", "data-designer-retrieval-sdg", "nemo-safe-synthesizer", "nemoguardrails", "nemo-compass", "nemo-switchyard", "nemo-fabric-runtime", "nemo-fabric-adapters-codex", "nemo-relay", "nvidia-nat-core", "nemo-new-library", "nemo-helix-tavily-plugin"];
  const table = render(names.map(name => pkg(name, "1.2.3")).join("\n"));
  for (const name of names) assert.ok(table.includes(`| \`${name}\` | \`1.2.3\` |`), name);
  assert.ok(table.includes("| Library / distribution | Version / source revision | Source of truth |"));
  assert.ok(!table.includes("Environment"));
  for (const row of table.split("\n").filter(line => line.startsWith("|"))) {
    assert.equal(row.split("|").length, 5, "the table should contain exactly three columns");
  }
});

test("excludes local plugin versions and unrelated runtime dependencies", () => {
  const table = render([
    pkg("nemo-anonymizer-plugin", "0.0.0", '{ editable = "plugins/nemo-anonymizer" }'),
    pkg("data-designer-nemo", "0.1.0", '{ directory = "packages/data_designer_nemo" }'),
    pkg("nemo-helix", "0.0.0", '{ virtual = "packages/nemo_helix" }'),
    pkg("nvidia-cublas", "999.0"), pkg("pydantic", "888.0"), pkg("nemo-anonymizer", "1.0"),
  ].join("\n"));
  for (const excluded of ["nemo-anonymizer-plugin", "data-designer-nemo", "nemo-helix", "nvidia-cublas", "pydantic"]) assert.ok(!table.includes(`\`${excluded}\``));
  assert.ok(table.includes("`nemo-anonymizer`"));
});

test("uses actual locked versions, tracks changes, and retains distinct resolutions", () => {
  const initial = render(pkg("nemo-relay", "1.0"));
  const updated = render(pkg("nemo-relay", "2.0") + pkg("nemo-relay", "2.1") + pkg("nemo-relay", "2.0"));
  assert.ok(initial.includes("`1.0`"));
  assert.ok(!updated.includes("`1.0`"));
  assert.ok(updated.includes("`2.0`, `2.1`"));
  assert.equal(updated.match(/\| `nemo-relay`/g).length, 1);
});

test("preserves Git revisions rather than presenting metadata versions as release pins", () => {
  const table = render(pkg("nemo-example", "1.0", '{ git = "https://example.com/nemo?rev=branch#abc123" }'));
  assert.ok(table.includes("`1.0 (Git abc123)`"));
});

test("discovers isolated locks and distinguishes their versions from workspace dependencies", () => {
  const paths = libraryVersionSourcePaths(["docker/locks/nhx-gym-host/uv.lock", "docker/locks/nhx-agent-hardener-tasks/uv.lock", "docker/locks/new-image/uv.lock", "docker/locks/new-image/pyproject.toml"]);
  assert.deepEqual(paths, ["docker/locks/new-image/uv.lock", "docker/locks/nhx-agent-hardener-tasks/uv.lock", "docker/locks/nhx-gym-host/uv.lock", "uv.lock"]);
  const table = render(pkg("nemo-relay", "1.0"), [
    ["docker/locks/nhx-agent-hardener-tasks/uv.lock", pkg("nvidia-agent-hardener", "3.0") + pkg("nemo-relay", "0.9")],
    ["docker/locks/nhx-gym-host/uv.lock", pkg("nemo-gym", "4.0")],
  ]);
  assert.ok(table.includes("`nvidia-agent-hardener` | `3.0`"));
  assert.ok(table.includes("`nemo-gym` | `4.0`"));
  assert.ok(table.includes("`nemo-relay` | `0.9` | `docker/locks/nhx-agent-hardener-tasks/uv.lock`"));
  assert.ok(table.includes("`nemo-relay` | `1.0` | `uv.lock`"));
});

test("reads Garak, AutoModel and RL source pins without inventing package release versions", () => {
  const table = render(pkg("nemo-example", "1.0"), [
    ["docker/Dockerfile.nhx-garak-tasks", "RUN uv pip install garak==9.8.7\n"],
    ["docker/automodel/Dockerfile.nhx-automodel-base", "ARG AUTOMODEL_BRANCH=r3.2.1\nARG AUTOMODEL_COMMIT=deadbeef\n"],
    ["docker-bake.hcl", 'variable "NEMO_RL_REF" {\n  default = "abcdef"\n}\n'],
  ]);
  assert.ok(table.includes("`garak` | `9.8.7`"));
  assert.ok(table.includes("`nemo-automodel` | `r3.2.1 (Git deadbeef)`"));
  assert.ok(table.includes("`nemo-rl` | `Git abcdef`"));
  assert.ok(table.includes("`nemo-gym` | `Submodule of NeMo-RL at abcdef`"));
});

test("fails loudly when version sources are absent, malformed or ambiguous", () => {
  assert.throws(() => buildLibraryVersionsMdx(new Map()), /uv.lock/);
  assert.throws(() => render("not TOML"));
  assert.throws(() => render("version = 1\n"), /no locked packages/);
  assert.throws(() => render('[[package]]\nname = "nemo-example"\n'), /no locked version/);
  assert.throws(() => render(pkg("nemo-example", "1.0"), [["docker/Dockerfile.nhx-garak-tasks", "RUN uv pip install garak\n"]]), /Expected one Garak/);
  assert.throws(() => render(pkg("nemo-example", "1.0"), [["docker/Dockerfile.nhx-garak-tasks", "RUN uv pip install garak==1.0 garak==2.0\n"]]), /found 2/);
});

test("does not swallow non-missing-file errors when reading generated output", t => {
  const root = mkdtempSync(join(tmpdir(), "helix-library-versions-error-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  writeFileSync(join(root, "uv.lock"), pkg("nemo-example", "1.0"));
  mkdirSync(join(root, "docs/fern/snippets/_snippets/library-versions.mdx"), { recursive: true });
  assert.throws(() => syncLibraryVersions({ root }), { code: "EISDIR" });
});

test("generation is deterministic and --check detects stale output", t => {
  const root = mkdtempSync(join(tmpdir(), "helix-library-versions-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const put = (path, text) => { mkdirSync(dirname(join(root, path)), { recursive: true }); writeFileSync(join(root, path), text); };
  const output = join(root, "docs/fern/snippets/_snippets/library-versions.mdx");
  put("uv.lock", pkg("nemo-example", "1.0"));
  assert.throws(() => syncLibraryVersions({ root, check: true }), /stale/);
  syncLibraryVersions({ root });
  const original = readFileSync(output, "utf8");
  syncLibraryVersions({ root, check: true });
  assert.equal(readFileSync(output, "utf8"), original);
  put("uv.lock", pkg("nemo-example", "2.0"));
  assert.throws(() => syncLibraryVersions({ root, check: true }), /stale/);
  syncLibraryVersions({ root });
  assert.ok(readFileSync(output, "utf8").includes("`2.0`"));
});
