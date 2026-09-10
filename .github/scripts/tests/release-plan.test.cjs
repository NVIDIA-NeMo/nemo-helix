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
  {
    id: "nemo-sandboxed-gym",
    package: "nemo-sandboxed-gym",
    path: "packages/sandboxed_gym",
    independent: true,
    tagPrefix: "nemo-sandboxed-gym-v",
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

test("bulk scopes leave an independent wheel alone", async () => {
  // The failure this guards is silent: a scheduled or whole-platform release quietly publishing
  // a version of nemo-sandboxed-gym that nothing pinned, on the platform's version line.
  for (const scope of ["all", "wheels"]) {
    const plan = await resolveReleasePlan({
      env: environment(),
      context: manualContext({
        "release-type": "stable",
        "release-scope": scope,
        "source-sha": SHA,
        version: "1.2.3",
      }),
      listBranches: async () =>
        assert.fail("stable releases do not discover release branches"),
    });

    assert.deepEqual(plan.wheelIds, ["nemo-platform"]);
  }
});

test("an independent wheel is still selectable by id", async () => {
  const plan = await resolveReleasePlan({
    env: environment(),
    context: manualContext({
      "release-type": "stable",
      "release-scope": "custom",
      "wheel-ids": "nemo-sandboxed-gym",
      "source-sha": SHA,
      version: "0.1.0",
    }),
    listBranches: async () =>
      assert.fail("stable releases do not discover release branches"),
  });

  // Alone: releasing it must not drag the platform wheels along.
  assert.deepEqual(plan.wheelIds, ["nemo-sandboxed-gym"]);
});

test("a pre-release version is accepted and flagged", async () => {
  const plan = await resolveReleasePlan({
    env: environment(),
    context: manualContext({
      "release-type": "stable",
      "release-scope": "custom",
      "wheel-ids": "nemo-sandboxed-gym",
      "source-sha": SHA,
      version: "0.1.0-rc0",
    }),
    listBranches: async () =>
      assert.fail("stable releases do not discover release branches"),
  });

  assert.equal(plan.isPrerelease, true);
  assert.equal(plan.releaseLabel, "0.1.0-rc0");
  assert.deepEqual(plan.wheelIds, ["nemo-sandboxed-gym"]);
});

test("a finished release is not flagged as a pre-release", async () => {
  const plan = await resolveReleasePlan({
    env: environment(),
    context: manualContext({
      "release-type": "stable",
      "release-scope": "wheels",
      "source-sha": SHA,
      version: "1.2.3",
    }),
    listBranches: async () =>
      assert.fail("stable releases do not discover release branches"),
  });

  assert.equal(plan.isPrerelease, false);
});

test("rejects a version whose pre-release suffix is not PEP 440 spellable", async () => {
  for (const version of [
    "1.2.3-alpha1",
    "1.2.3-rc",
    "1.2.3-rc0.1",
    "1.2.3rc0",
  ]) {
    await assert.rejects(
      resolveReleasePlan({
        env: environment(),
        context: manualContext({
          "release-type": "stable",
          "release-scope": "wheels",
          "source-sha": SHA,
          version,
        }),
        listBranches: async () =>
          assert.fail("stable releases do not discover release branches"),
      }),
      /MAJOR\.MINOR\.PATCH/,
      `expected ${version} to be rejected`,
    );
  }
});

test("an independent wheel cannot ride along with other artifacts", async () => {
  const selections = [
    { "wheel-ids": "nemo-platform,nemo-sandboxed-gym" },
    { "wheel-ids": "nemo-sandboxed-gym", "container-ids": "nmp-api" },
    { "wheel-ids": "nemo-sandboxed-gym", "include-helm": "true" },
  ];
  for (const extra of selections) {
    await assert.rejects(
      resolveReleasePlan({
        env: environment(),
        context: manualContext({
          "release-type": "stable",
          "release-scope": "custom",
          "source-sha": SHA,
          version: "1.2.3",
          ...extra,
        }),
        listBranches: async () =>
          assert.fail("stable releases do not discover release branches"),
      }),
      /independent wheel must be released on its own/,
      `expected ${JSON.stringify(extra)} to be rejected`,
    );
  }
});

test("an independent wheel alone is still a valid release", async () => {
  const plan = await resolveReleasePlan({
    env: environment(),
    context: manualContext({
      "release-type": "stable",
      "release-scope": "custom",
      "wheel-ids": "nemo-sandboxed-gym",
      "source-sha": SHA,
      version: "0.1.0-rc0",
    }),
    listBranches: async () =>
      assert.fail("stable releases do not discover release branches"),
  });

  assert.deepEqual(plan.wheelIds, ["nemo-sandboxed-gym"]);
  assert.equal(plan.isPrerelease, true);
});

test("an independent wheel tags under its own prefix", async () => {
  const plan = await resolveReleasePlan({
    env: environment(),
    context: manualContext({
      "release-type": "stable",
      "release-scope": "custom",
      "wheel-ids": "nemo-sandboxed-gym",
      "source-sha": SHA,
      version: "0.1.0-rc0",
    }),
    getCommit: async () => assert.fail("stable releases pin their source"),
  });

  // Not `0.1.0-rc0`: the platform already owns tags in that namespace, and the package's
  // dynamic versioning reads this prefix.
  assert.equal(plan.releaseTag, "nemo-sandboxed-gym-v0.1.0-rc0");
  assert.equal(plan.tagPrefix, "nemo-sandboxed-gym-v");
});

test("a platform release still tags unprefixed", async () => {
  const plan = await resolveReleasePlan({
    env: environment(),
    context: manualContext({
      "release-type": "stable",
      "release-scope": "wheels",
      "source-sha": SHA,
      version: "1.2.3",
    }),
    getCommit: async () => assert.fail("stable releases pin their source"),
  });

  assert.equal(plan.releaseTag, "1.2.3");
  assert.equal(plan.tagPrefix, "");
});

test("an independent wheel cannot be released as a nightly", async () => {
  await assert.rejects(
    resolveReleasePlan({
      env: environment(),
      context: manualContext({
        "release-type": "nightly",
        "release-scope": "custom",
        "wheel-ids": "nemo-sandboxed-gym",
        "dry-run": "true",
      }),
      listBranches: async () => BRANCHES,
    }),
    /versioned from its own tags/,
  );
});
