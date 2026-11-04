// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const path = require("node:path");
const process = require("node:process");
const test = require("node:test");
const script = require.resolve("../configure-docker-cache.sh");

function configure(event, ref, scope = "cpu") {
  const result = spawnSync("bash", [script], {
    encoding: "utf8",
    env: {
      PATH: process.env.PATH,
      GITHUB_REPOSITORY: "NVIDIA-NeMo/nemo-helix",
      GITHUB_EVENT_NAME: event,
      GITHUB_REF: ref,
      CACHE_SCOPE: scope,
      GITHUB_ENV: "/dev/stdout",
      GITHUB_STEP_SUMMARY: "/dev/null",
      BAKE_CACHE_TARGET_BRANCH: "main", // Consumers must clear inherited writes.
    },
    timeout: 10_000,
  });
  assert.equal(result.status, 0, result.stderr);
  return Object.fromEntries(
    result.stdout
      .trim()
      .split("\n")
      .map((line) => line.split("=")),
  );
}

test("only main pushes export; image jobs use separate cache namespaces", () => {
  for (const scope of ["cpu", "garak", "safe-synthesizer"]) {
    const config = configure("push", "refs/heads/main", scope);
    assert.equal(config.BAKE_CACHE_TARGET_BRANCH, "main");
    assert.equal(
      config.CACHE_REGISTRY,
      `ghcr.io/nvidia-nemo/nemo-helix/ci-cache/${scope}`,
    );
  }
  for (const [event, ref] of [
    ["pull_request", "refs/pull/123/merge"],
    ["merge_group", "refs/heads/gh-readonly-queue/main/pr-123-abc"],
    ["workflow_dispatch", "refs/heads/main"],
    ["push", "refs/heads/release/0.9"],
  ]) {
    const config = configure(event, ref);
    assert.equal(config.BAKE_CACHE_SOURCE_BRANCH, "main");
    assert.equal(config.BAKE_CACHE_TARGET_BRANCH, "", `${event}: ${ref}`);
  }
});

// Evaluate the real HCL without a daemon, build, or registry.
const hasBuildx =
  spawnSync("docker", ["buildx", "version"], { timeout: 10_000 }).status === 0;
test(
  "CPU Bake graph caches both the API image and its base",
  { skip: !hasBuildx },
  () => {
    for (const event of ["push", "merge_group"]) {
      const config = configure(event, "refs/heads/main");
      const result = spawnSync(
        "docker",
        ["buildx", "bake", "--print", "docker-cpu-ci"],
        {
          cwd: path.resolve(path.dirname(script), "../.."),
          encoding: "utf8",
          env: {
            ...process.env,
            ...config,
            BUILD_ARCH: "linux/amd64",
            CACHE_VERSION: "cache",
          },
          timeout: 10_000,
        },
      );
      assert.equal(result.status, 0, result.stderr);
      const targets = JSON.parse(result.stdout).target;
      for (const [target, image] of [
        ["nhx-api-docker", "nhx-api"],
        ["nhx-python-base", "nhx-python-base"],
      ]) {
        const ref = `${config.CACHE_REGISTRY}/${image}:cache-main-linux-amd64`;
        assert.ok(
          targets[target]["cache-from"].some((cache) => cache.ref === ref),
        );
        const exports = targets[target]["cache-to"] || [];
        if (event === "push") {
          assert.equal(exports.length, 1);
          assert.equal(exports[0].ref, ref);
          assert.equal(exports[0].mode, "max");
          assert.equal(exports[0]["ignore-error"], "true");
        } else {
          assert.deepEqual(exports, []);
        }
      }
    }
  },
);
