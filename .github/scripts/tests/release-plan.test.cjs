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
const BRANCHES = [
  { name: "release/0.10", commit: { sha: SHA } },
  { name: "release/0.9", commit: { sha: "c".repeat(40) } },
  { name: "main", commit: { sha: "b".repeat(40) } },
];

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
    listBranches: async () =>
      assert.fail("stable releases do not discover release branches"),
  });

  assert.equal(plan.sourceSha, SHA);
  assert.equal(plan.sourceBranch, "");
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
    listBranches: async () => BRANCHES,
    now: () => new Date("2026-08-27T12:34:56.789Z"),
  });

  assert.equal(plan.sourceSha, SHA);
  assert.equal(plan.sourceBranch, "release/0.10");
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
      listBranches: async () => BRANCHES,
    }),
    /wheel-ids contains duplicate entries/,
  );
});

for (const [label, context] of [
  ["scheduled", { eventName: "schedule", sha: "b".repeat(40), payload: {} }],
  ["manual", manualContext({ "release-type": "nightly" })],
  ["dry-run", manualContext({ "release-type": "nightly", "dry-run": "true" })],
]) {
  test(`${label} nightlies select the highest release branch and pin its SHA`, async () => {
    const plan = await resolveReleasePlan({
      env: environment(),
      context,
      listBranches: async () => BRANCHES,
    });

    assert.equal(plan.sourceBranch, "release/0.10");
    assert.equal(plan.sourceSha, SHA);
  });
}

test("compares major versions before minor versions and ignores unrelated branches", async () => {
  const plan = await resolveReleasePlan({
    env: environment(),
    context: manualContext({ "release-type": "nightly" }),
    listBranches: async () => [
      { name: "release/2.99", commit: { sha: "c".repeat(40) } },
      { name: "release/10.0", commit: { sha: SHA } },
      ...[
        "main",
        "release/99",
        "release/99.0.0",
        "release/99.0-rc1",
        "release/99.0/fix",
        "feature/release/99.0",
      ].map((name) => ({ name, commit: { sha: "d".repeat(40) } })),
    ],
  });

  assert.equal(plan.sourceBranch, "release/10.0");
  assert.equal(plan.sourceSha, SHA);
});

for (const dryRun of ["false", "true"]) {
  test(`explicit nightly SHAs bypass branch discovery (dry-run=${dryRun})`, async () => {
    const plan = await resolveReleasePlan({
      env: environment(),
      context: manualContext({
        "release-type": "nightly",
        "source-sha": ` ${SHA.toUpperCase()} `,
        "dry-run": dryRun,
      }),
      listBranches: async () =>
        assert.fail("an explicit SHA bypasses branch discovery"),
    });

    assert.equal(plan.sourceSha, SHA);
    assert.equal(plan.sourceBranch, "");
  });
}

for (const branches of [[], [BRANCHES[2]]]) {
  test(`fails without release branches (${branches.length} total branches)`, async () => {
    await assert.rejects(
      resolveReleasePlan({
        env: environment(),
        context: manualContext({ "release-type": "nightly" }),
        listBranches: async () => branches,
      }),
      /No release\/X\.X branch found/,
    );
  });
}

test("propagates branch lookup failures instead of falling back to the workflow SHA", async () => {
  const error = new Error("GitHub branch lookup failed");
  await assert.rejects(
    resolveReleasePlan({
      env: environment(),
      context: manualContext({ "release-type": "nightly" }),
      listBranches: async () => {
        throw error;
      },
    }),
    error,
  );
});

for (const releaseType of ["nightly", "stable"]) {
  test(`rejects invalid explicit ${releaseType} SHAs before branch discovery`, async () => {
    await assert.rejects(
      resolveReleasePlan({
        env: environment(),
        context: manualContext({
          "release-type": releaseType,
          "source-sha": "release/0.10",
          version: "0.10.0",
        }),
        listBranches: async () =>
          assert.fail("invalid SHAs must fail before branch discovery"),
      }),
      /exact 40-character.*SHA/,
    );
  });
}
