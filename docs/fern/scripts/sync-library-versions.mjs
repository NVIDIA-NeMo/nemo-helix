// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Generate the support-matrix library table from lockfiles and image pins. */
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "smol-toml";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
export const libraryVersionsInclude = '<Markdown src="/snippets/_snippets/library-versions.mdx" />';
export const imageVersionSources = [
  "docker-bake.hcl",
  "docker/automodel/Dockerfile.nhx-automodel-base",
  "docker/Dockerfile.nhx-garak-tasks",
];
const snippetPath = "docs/fern/snippets/_snippets/library-versions.mdx";

function canonicalName(name) {
  return name.toLowerCase().replace(/[-_.]+/g, "-");
}

function isLibrary(name) {
  return name.startsWith("nemo-")
    || name.startsWith("data-designer")
    || name.startsWith("nvidia-nat-")
    || ["nemoguardrails", "garak", "nvidia-agent-hardener"].includes(name);
}

function cell(text) {
  return String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll("{", "&#123;")
    .replaceAll("|", "&#124;").replaceAll("`", "&#96;").replace(/\r?\n/g, " ");
}

function code(text) {
  return `\`${cell(text)}\``;
}

function onePin(text, pattern, label) {
  const matches = [...new Set([...text.matchAll(pattern)].map(match => match[1]))];
  if (matches.length !== 1) {
    throw new Error(`Expected one ${label} pin, found ${matches.length}. Update the library-version source reader.`);
  }
  return matches[0];
}

/** Files needed from a checkout or a release Git tree; no installed packages are queried. */
export function libraryVersionSourcePaths(paths) {
  return [...new Set([
    "uv.lock",
    ...paths.filter(path => /^docker\/locks\/.+\/uv\.lock$/.test(path)),
    ...imageVersionSources.filter(path => paths.includes(path)),
  ])].sort();
}

export function buildLibraryVersionsMdx(sources) {
  if (!sources.has("uv.lock")) throw new Error("Library versions require the source checkout's uv.lock");
  const rows = [];
  for (const [path, text] of [...sources].sort(([a], [b]) => a.localeCompare(b))) {
    if (!path.endsWith("uv.lock")) continue;
    const packages = parse(text).package;
    if (!Array.isArray(packages)) throw new Error(`${path} has no locked packages`);
    const versions = new Map();
    for (const pkg of packages) {
      const name = canonicalName(pkg.name);
      // Helix's in-tree plugin/service versions are not upstream library versions.
      if (!isLibrary(name) || pkg.source?.editable || pkg.source?.virtual || pkg.source?.directory) continue;
      if (!pkg.version) throw new Error(`${path}: ${name} has no locked version`);
      const revision = pkg.source?.git?.split("#")[1];
      const version = pkg.version + (revision ? ` (Git ${revision})` : "");
      const values = versions.get(name) ?? new Set();
      values.add(version);
      versions.set(name, values);
    }
    for (const [name, values] of versions) {
      const environment = path === "uv.lock" ? "Helix workspace (including optional dependencies)"
        : `Isolated image: ${path.split("/").at(-2)}`;
      rows.push({ name, versions: [...values].sort(), environment, path });
    }
  }

  const garakPath = "docker/Dockerfile.nhx-garak-tasks";
  if (sources.has(garakPath)) {
    const version = onePin(sources.get(garakPath), /\bgarak==([\w.!+-]+)/g, "Garak audit-image");
    rows.push({ name: "garak", versions: [version], environment: "Garak audit image; Agent Hardener provisions its own Garak environment", path: garakPath });
  }
  const automodelPath = "docker/automodel/Dockerfile.nhx-automodel-base";
  if (sources.has(automodelPath)) {
    const text = sources.get(automodelPath);
    const branch = onePin(text, /^ARG AUTOMODEL_BRANCH=([^\s#]+)/gm, "AutoModel release-branch");
    const revision = onePin(text, /^ARG AUTOMODEL_COMMIT=([^\s#]+)/gm, "AutoModel commit");
    rows.push({ name: "nemo-automodel", versions: [`${branch} (Git ${revision})`], environment: "AutoModel training base", path: automodelPath });
  }
  if (sources.has("docker-bake.hcl")) {
    const text = sources.get("docker-bake.hcl");
    const revision = onePin(text, /variable "NEMO_RL_REF"\s*\{\s*default\s*=\s*"([^"]+)"/g, "NeMo-RL source revision");
    rows.push({ name: "nemo-rl", versions: [`Git ${revision}`], environment: "RL training base", path: "docker-bake.hcl" });
    rows.push({ name: "nemo-gym", versions: [`Submodule of NeMo-RL at ${revision}`], environment: "RL training base; distinct from the isolated Gym runner locks", path: "docker-bake.hcl" });
  }

  rows.sort((a, b) => a.name.localeCompare(b.name) || a.path.localeCompare(b.path));
  return [
    "{/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */}",
    "{/* SPDX-License-Identifier: Apache-2.0 */}",
    "",
    "{/* Generated by sync-library-versions.mjs; edit dependency or image pins, not this table. */}",
    "",
    "| Library / distribution | Version / source revision | Environment | Source of truth |",
    "| --- | --- | --- | --- |",
    ...rows.map(row => `| ${code(row.name)} | ${row.versions.map(code).join(", ")} | ${cell(row.environment)} | ${code(row.path)} |`),
    "",
  ].join("\n");
}

function checkoutPaths(dir, root = dir) {
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const path = join(dir, entry.name);
    return entry.isDirectory() ? checkoutPaths(path, root) : [relative(root, path).replaceAll("\\", "/")];
  });
}

export function syncLibraryVersions({ root = repoRoot, check = false } = {}) {
  const paths = [
    ...checkoutPaths(join(root, "docker/locks"), root),
    ...imageVersionSources.filter(path => existsSync(join(root, path))),
  ];
  const sources = new Map(libraryVersionSourcePaths(paths).map(path => [path, readFileSync(join(root, path), "utf8")]));
  const generated = buildLibraryVersionsMdx(sources);
  const output = join(root, snippetPath);
  const current = existsSync(output) ? readFileSync(output, "utf8") : "";
  if (current === generated) return;
  if (check) throw new Error("Library-version table is stale. Run npm run prepare:libraries in docs/fern.");
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, generated);
  console.log(`[sync-library-versions] Updated ${snippetPath}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  if (args.some(arg => arg !== "--check")) throw new Error("Only --check is supported");
  syncLibraryVersions({ check: args.includes("--check") });
}
