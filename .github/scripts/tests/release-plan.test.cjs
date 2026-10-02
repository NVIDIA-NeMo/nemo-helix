// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for release plan parsing.
const assert = require("node:assert/strict");
const test = require("node:test");

const { resolveReleasePlan } = require("../release-plan.cjs");

const WHEELS = [
  {
    id: "nemo-platform",
    package: "nemo-platform",
    path: "packages/nemo_platform",
  },
];
const CONTAINERS = [{ id: "nmp-api", target: "nmp-api-docker" }];
const SHA = "a".repeat(40);

function environment() {
  return {
    RELEASE_WHEELS_JSON: JSON.stringify(WHEELS),
    RELEASE_CONTAINERS_JSON: JSON.stringify(CONTAINERS),
  };
}

function manualContext(inputs) {
  return {
    eventName: "workflow_dispatch",
    sha: "b".repeat(40),
    payload: { inputs, repository: { default_branch: "main" } },
  };
}

test("resolves a stable Helm-only release", async () => {
  const plan = await resolveReleasePlan({
    env: environment(),
    context: manualContext({
      "release-type": "stable",
      "release-scope": "helm",
      "source-sha": SHA.toUpperCase(),
      version: "1.2.3",
      "helm-version": "1.2.3-rc.1+build.9",
    }),
    getCommit: async () =>
      assert.fail("stable releases do not resolve a default branch commit"),
  });

  assert.equal(plan.sourceSha, SHA);
  assert.equal(plan.releaseLabel, "1.2.3");
  assert.equal(plan.includeHelm, true);
  assert.deepEqual(plan.wheelIds, []);
  assert.deepEqual(plan.containerIds, []);
});

test("resolves a custom nightly release and uses the supplied clock", async () => {
  const plan = await resolveReleasePlan({
    env: environment(),
    context: manualContext({
      "release-type": "nightly",
      "release-scope": "custom",
      "wheel-ids": "nemo-platform",
      "container-ids": "nmp-api",
      "include-helm": "false",
      "dry-run": "true",
    }),
    getCommit: async () => assert.fail("dry runs use the workflow SHA"),
    now: () => new Date("2026-08-27T12:34:56.789Z"),
  });

  assert.equal(plan.sourceSha, "b".repeat(40));
  assert.equal(plan.releaseLabel, "nightly-20260827123456");
  assert.deepEqual(plan.wheelIds, ["nemo-platform"]);
  assert.deepEqual(plan.containerIds, ["nmp-api"]);
});

test("rejects duplicate custom artifact IDs", async () => {
  await assert.rejects(
    resolveReleasePlan({
      env: environment(),
      context: manualContext({
        "release-type": "nightly",
        "release-scope": "custom",
        "wheel-ids": "nemo-platform,nemo-platform",
      }),
      getCommit: async () => SHA,
    }),
    /wheel-ids contains duplicate entries/,
  );
});

for (const branch of ["release/0.5", "release/0.6"]) {
  for (const pinned of [false, true]) {
    test(`${branch} nightly uses ${pinned ? "the pinned source" : "the workflow commit"}`, async () => {
      const context = manualContext({
        "release-type": "nightly",
        "source-sha": pinned ? SHA : "",
      });
      context.ref = `refs/heads/${branch}`;
      const plan = await resolveReleasePlan({
        env: environment(),
        context,
        getCommit: async () =>
          assert.fail(
            "Do not select the default branch for a maintenance nightly",
          ),
      });
      assert.equal(plan.sourceSha, pinned ? SHA : context.sha);
    });
  }
}

test("an unpinned nightly on main retains default-branch selection", async () => {
  const context = manualContext({ "release-type": "nightly" });
  context.ref = "refs/heads/main";
  const plan = await resolveReleasePlan({
    env: environment(),
    context,
    getCommit: async (ref) => {
      assert.equal(ref, "main");
      return SHA;
    },
  });
  assert.equal(plan.sourceSha, SHA);
});
