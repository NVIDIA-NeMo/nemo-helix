#!/usr/bin/env node
/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Ensure every NotebookViewer import is registered, every registration has
 * generated notebook data on disk, and that the generated JSON and TypeScript
 * artifacts still match their source notebooks.
 *
 * NotebookViewer.tsx imports `./notebooks/<name>` modules produced by
 * `ipynb-to-fern-json.py` and looks them up by name in `const notebooks`.
 * An import missing from that registry cannot be resolved by
 * `<NotebookViewer name="…" />`. A missing `.ts` / `.json` pair breaks
 * publication even when the MDX wrapper and registry entry exist.
 *
 * Run from the fern/ directory: `node scripts/validate-notebook-viewer.mjs`.
 */

import { access, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const VIEWER = join(ROOT, "../components/NotebookViewer.tsx");
const NOTEBOOKS_DIR = join(ROOT, "../components/notebooks");
const SOURCE_NOTEBOOKS = {
  "distillation-customization-job": join(
    ROOT,
    "../../customizer/tutorials/distillation-customization-job.ipynb",
  ),
  "dpo-customization-job": join(
    ROOT,
    "../../customizer/tutorials/dpo-customization-job.ipynb",
  ),
  "embedding-customization-job": join(
    ROOT,
    "../../customizer/tutorials/embedding-customization-job.ipynb",
  ),
  "embedding-triplet-dataset": join(
    ROOT,
    "../../customizer/tutorials/embedding-triplet-dataset.ipynb",
  ),
  "lora-customization-job": join(
    ROOT,
    "../../customizer/tutorials/lora-customization-job.ipynb",
  ),
  "optimize-throughput": join(
    ROOT,
    "../../customizer/tutorials/optimize-throughput.ipynb",
  ),
  "retrieval-generate": join(
    ROOT,
    "../../data-designer/tutorials/retrieval-generate.ipynb",
  ),
  "sft-customization-job": join(
    ROOT,
    "../../customizer/tutorials/sft-customization-job.ipynb",
  ),
  "tool-calling": join(ROOT, "../../example-applications/tool-calling.ipynb"),
};

const IMPORT_RE =
  /import\s+\w+\s+from\s+"\.\/notebooks\/([^"]+)";/g;
const REGISTRY_RE = /"([^"]+)":\s*\w+/g;

const viewerSrc = await readFile(VIEWER, "utf8");
const imported = [...viewerSrc.matchAll(IMPORT_RE)].map((m) => m[1]);
const registryBlock = viewerSrc.match(
  /const notebooks: Record<string, unknown> = \{([\s\S]*?)\};/
)?.[1];

if (!registryBlock) {
  console.error("validate-notebook-viewer: could not find notebooks registry");
  process.exit(2);
}

const registered = [...registryBlock.matchAll(REGISTRY_RE)].map((m) => m[1]);
const registeredSet = new Set(registered);
const unregisteredImports = [...new Set(imported)]
  .filter((name) => !registeredSet.has(name))
  .sort();

if (unregisteredImports.length > 0) {
  for (const name of unregisteredImports) {
    console.error(
      `imported ./notebooks/${name} but missing from NotebookViewer notebooks registry`,
    );
  }
  console.error(
    `\nvalidate-notebook-viewer: ${unregisteredImports.length} imported notebook(s) are not registered. ` +
      `Add each name to the const notebooks map in NotebookViewer.tsx.`,
  );
  process.exit(1);
}

// Only registered notebooks are resolvable via <NotebookViewer name="…" />.
const names = [...registeredSet].sort();

let failed = 0;
for (const name of names) {
  for (const ext of ["ts", "json"]) {
    const path = join(NOTEBOOKS_DIR, `${name}.${ext}`);
    try {
      await access(path);
    } catch {
      failed += 1;
      console.error(`missing ${path}`);
    }
  }

  const sourcePath = SOURCE_NOTEBOOKS[name];
  if (!sourcePath) {
    failed += 1;
    console.error(`missing source-notebook mapping for ${name}`);
    continue;
  }

  try {
    const notebook = JSON.parse(await readFile(sourcePath, "utf8"));
    const jsonPath = join(NOTEBOOKS_DIR, `${name}.json`);
    const tsPath = join(NOTEBOOKS_DIR, `${name}.ts`);
    const artifact = JSON.parse(await readFile(jsonPath, "utf8"));
    const sourceCells = notebook.cells.map((cell) =>
      (Array.isArray(cell.source) ? cell.source.join("") : cell.source ?? "").trimEnd(),
    );
    const artifactCells = artifact.cells.map((cell) =>
      (cell.source ?? "").trimEnd(),
    );

    const mismatch =
      sourceCells.length !== artifactCells.length ||
      sourceCells.some((source, index) => source !== artifactCells[index]);
    if (mismatch) {
      failed += 1;
      console.error(
        `stale ${jsonPath}; regenerate it from ${sourcePath}`,
      );
    }

    const tsSource = await readFile(tsPath, "utf8");
    const tsMatch = tsSource.match(/export\s+default\s+\{\s*cells:\s*(\[[\s\S]*\])\s*\};\s*$/);
    if (!tsMatch) {
      failed += 1;
      console.error(`could not parse default export in ${tsPath}`);
    } else {
      const tsCells = JSON.parse(tsMatch[1]).map((cell) =>
        (cell.source ?? "").trimEnd(),
      );
      const tsMismatch =
        artifactCells.length !== tsCells.length ||
        artifactCells.some((source, index) => source !== tsCells[index]);
      if (tsMismatch) {
        failed += 1;
        console.error(
          `stale ${tsPath}; regenerate it from ${sourcePath}`,
        );
      }
    }
  } catch (error) {
    failed += 1;
    console.error(`could not compare ${name} with ${sourcePath}: ${error.message}`);
  }
}

if (failed > 0) {
  console.error(
    `\nvalidate-notebook-viewer: ${failed} missing or stale notebook artifact(s). ` +
      `Run: uv run python docs/fern/scripts/ipynb-to-fern-json.py <notebook.ipynb> ` +
      `-o docs/fern/components/notebooks/<name>.json`
  );
  process.exit(1);
}

console.log(
  `validate-notebook-viewer: ${names.length} NotebookViewer notebook(s) present`
);
