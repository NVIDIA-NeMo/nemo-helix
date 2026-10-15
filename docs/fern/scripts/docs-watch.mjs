// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawn } from "node:child_process";
import { watch } from "node:fs";
import { utimes } from "node:fs/promises";
import { constants } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { syncHelmDocs } from "./sync-helm-docs.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const fernDir = path.resolve(scriptDir, "..");
const docsRoot = path.resolve(fernDir, "..");
const repoRoot = path.resolve(docsRoot, "..");
const helmDir = path.join(repoRoot, "k8s", "helm");
const reloadTrigger = path.join(fernDir, "docs.yml");
const reloadDebounceMs = 150;
const ignoredPrefix = "fern/";
const openapiInputPath = "fern/openapi/openapi.yaml";
const helmWatchFiles = new Set(["values.yaml", "README.md"]);

let debounceTimer = null;
let shuttingDown = false;

function log(message) {
  process.stdout.write(`[docs-watch] ${message}\n`);
}

async function touchReloadTrigger(changedPath) {
  const now = new Date();
  await utimes(reloadTrigger, now, now);
  log(`triggered Fern reload via docs.yml after change in ${changedPath}`);
}

function clearPendingReload() {
  if (debounceTimer === null) {
    return;
  }
  clearTimeout(debounceTimer);
  debounceTimer = null;
}

function scheduleReload(relativePath) {
  clearPendingReload();

  debounceTimer = setTimeout(() => {
    debounceTimer = null;
    touchReloadTrigger(relativePath).catch((error) => {
      log(`failed to trigger reload: ${error.message}`);
    });
  }, reloadDebounceMs);
}

function normalizeWatchedPath(relativePath) {
  return path.posix.normalize(relativePath.split(path.sep).join("/"));
}

function shouldIgnore(normalizedPath) {
  return normalizedPath.startsWith(ignoredPrefix) && normalizedPath !== openapiInputPath;
}

function prepareHelm() {
  syncHelmDocs();
}

function spawnFernDev() {
  prepareHelm();
  return spawn("npx", ["-y", "fern-api@latest", "docs", "dev"], {
    cwd: fernDir,
    stdio: "inherit",
  });
}

const fern = spawnFernDev();

const watcher = watch(
  docsRoot,
  { recursive: true },
  (_eventType, filename) => {
    const relativePath = filename ? filename.toString() : "";
    const normalizedPath = normalizeWatchedPath(relativePath);
    if (shouldIgnore(normalizedPath)) {
      return;
    }
    scheduleReload(relativePath);
  },
);

watcher.on("error", (error) => {
  log(`watcher error: ${error.message}`);
});

const helmWatcher = watch(helmDir, (_eventType, filename) => {
  if (!filename || !helmWatchFiles.has(filename.toString())) {
    return;
  }
  log(`helm source changed (${filename}), regenerating helm docs`);
  try {
    prepareHelm();
  } catch (error) {
    log(`failed to regenerate helm docs: ${error.message}`);
    return;
  }
  touchReloadTrigger(`k8s/helm/${filename}`).catch((error) => {
    log(`failed to trigger reload after helm change: ${error.message}`);
  });
});

helmWatcher.on("error", (error) => {
  log(`helm watcher error: ${error.message}`);
});

function closeWatcher() {
  watcher.close();
  helmWatcher.close();
}

function signalExitCode(signal) {
  const signalNumber = constants.signals[signal];
  return signalNumber ? 128 + signalNumber : 1;
}

function shutdown(signal) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  closeWatcher();
  clearPendingReload();
  fern.kill(signal);
  process.exit(signalExitCode(signal));
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

fern.on("exit", (code, signal) => {
  closeWatcher();
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
