// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for release plan parsing.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const test = require("node:test");

const {
  resolveReleasePlan,
  resolveRcBaseVersion,
  readReleaseCatalog,
} = require("../release-plan.cjs");

const WHEELS = [
  {
    id: "nemo-helix",
    package: "nemo-helix",
    path: "packages/nemo_helix",
  },
  {
    id: "nemo-sandboxed-gym",
    package: "nemo-sandboxed-gym",
    path: "packages/sandboxed_gym",
    independent: true,
    tagPrefix: "nemo-sandboxed-gym-v",
  },
];
const CONTAINERS = [{ id: "nhx-api", target: "nhx-api-docker" }];
const SHA = "a".repeat(40);
const BRANCHES = [
  { name: "release/0.10", commit: { sha: SHA } },
  { name: "release/0.9", commit: { sha: "c".repeat(40) } },
  { name: "main", commit: { sha: "b".repeat(40) } },
];

function sourceWorkflow({ wheels = WHEELS, containers = CONTAINERS } = {}) {
  return [
    "env:",
    "  RELEASE_WHEELS_JSON: >-",
    `    ${JSON.stringify(wheels)}`,
    "  RELEASE_CONTAINERS_JSON: >-",
    `    ${JSON.stringify(containers)}`,
    "  RELEASE_HELM_ID: nemo-helix",
    "",
  ].join("\n");
}

function manualContext(inputs) {
  return {
    eventName: "workflow_dispatch",
    sha: "b".repeat(40),
    payload: { inputs, repository: { default_branch: "main" } },
  };
}

test("rejects unsupported release types before reading source or dispatching", async () => {
  for (const releaseType of ["nightly", "unknown", "", undefined]) {
    await assert.rejects(
      resolveReleasePlan({
        context: manualContext({ "release-type": releaseType }),
        listBranches: () => assert.fail("must not discover branches"),
        loadSourceWorkflow: () => assert.fail("must not read a source catalog"),
      }),
      /Unsupported release type/,
    );
  }
});

test("reads the checked-in release catalog's multiline JSON blocks", () => {
  const workflow = fs.readFileSync(
    require.resolve("../../workflows/release.yaml"),
    "utf8",
  );
  for (const text of [workflow, workflow.replace(/\n/g, "\r\n")]) {
    const catalog = readReleaseCatalog(text);
    assert.ok(catalog.wheels.some(({ id }) => id === "nemo-helix"));
    assert.ok(catalog.containers.some(({ id }) => id === "nhx-api"));
  }
});

test("rejects missing or invalid source catalogs", () => {
  assert.throws(
    () => readReleaseCatalog("env: {}"),
    /missing RELEASE_WHEELS_JSON/,
  );
  assert.throws(
    () =>
      readReleaseCatalog(
        sourceWorkflow().replace("RELEASE_CONTAINERS_JSON", "OTHER_JSON"),
      ),
    /missing RELEASE_CONTAINERS_JSON/,
  );
  assert.throws(
    () => readReleaseCatalog(sourceWorkflow({ containers: {} })),
    /RELEASE_CONTAINERS_JSON must be a JSON array/,
  );
  for (const name of ["RELEASE_WHEELS_JSON", "RELEASE_CONTAINERS_JSON"]) {
    assert.throws(
      () =>
        readReleaseCatalog(
          sourceWorkflow().replace(`${name}: >-\n    [`, `${name}: >-\n    ![`),
        ),
      (error) => {
        assert.equal(error.constructor, Error);
        assert.equal(
          error.message,
          `Selected source ${name} contains invalid JSON.`,
        );
        assert.ok(error.cause instanceof SyntaxError);
        return true;
      },
    );
  }
});

for (const [label, context] of [
  ["scheduled", { eventName: "schedule", payload: {} }],
  ["manual", manualContext({ "release-type": "rc" })],
  ["dry-run", manualContext({ "release-type": "rc", "dry-run": "true" })],
  [
    "stable",
    manualContext({
      "release-type": "stable",
      "source-sha": SHA,
      version: "0.10.0",
    }),
  ],
]) {
  test(`${label} plans select artifacts from the source catalog instead of main`, async () => {
    const plan = await resolveReleasePlan({
      context,
      // A newer workflow revision ships images and wheels absent from the release source.
      env: {
        RELEASE_WHEELS_JSON: JSON.stringify([
          ...WHEELS,
          { id: "main-only-wheel" },
        ]),
        RELEASE_CONTAINERS_JSON: JSON.stringify([
          ...CONTAINERS,
          { id: "nhx-builder-tasks", target: "nhx-builder-tasks-docker" },
          { id: "nhx-kaniko", target: "nhx-kaniko-docker" },
          { id: "nhx-agent-installer", target: "nhx-agent-installer-docker" },
        ]),
      },
      listBranches: async () => BRANCHES,
      loadSourceWorkflow: async (sha) => {
        assert.equal(sha, SHA);
        return sourceWorkflow();
      },
    });
    assert.deepEqual(plan.wheelIds, ["nemo-helix"]);
    assert.deepEqual(plan.containerIds, ["nhx-api"]);
  });
}

test("pinned RCs read the pinned commit's catalog rather than the branch head", async () => {
  const pinnedSha = "d".repeat(40);
  const plan = await resolveReleasePlan({
    context: manualContext({
      "release-type": "rc",
      "source-sha": pinnedSha.toUpperCase(),
    }),
    listBranches: async () => BRANCHES,
    compareCommits: async () => ({ status: "ahead" }),
    loadSourceWorkflow: async (sha) => {
      assert.equal(sha, pinnedSha);
      return sourceWorkflow({ containers: [] });
    },
  });
  assert.deepEqual(plan.containerIds, []);
});

test("custom selections reject IDs not shipped by the source", async () => {
  await assert.rejects(
    resolveReleasePlan({
      context: manualContext({
        "release-type": "rc",
        "release-scope": "custom",
        "container-ids": "nhx-builder-tasks",
      }),
      listBranches: async () => BRANCHES,
      loadSourceWorkflow: async () => sourceWorkflow(),
    }),
    /Unknown container IDs: nhx-builder-tasks/,
  );
});

test("a source catalog read failure stops planning", async () => {
  const error = new Error("source workflow unavailable");
  await assert.rejects(
    resolveReleasePlan({
      context: { eventName: "schedule", payload: {} },
      listBranches: async () => BRANCHES,
      loadSourceWorkflow: async () => {
        throw error;
      },
    }),
    error,
  );
});

for (const [branch, tags, expected] of [
  ["release/0.6", [], "0.6.0"],
  ["release/0.6", ["0.5.0", "0.5.1", "0.7.9", "1.6.9"], "0.6.0"],
  ["release/0.6", ["0.6.0"], "0.6.1"],
  ["release/0.6", ["0.6.2", "0.6.0"], "0.6.3"],
  ["release/0.6", ["0.6.9", "0.6.10", "0.6.2"], "0.6.11"],
  [
    "release/0.6",
    ["0.6.0-rc0", "0.6.0-rc.1", "v0.6.5", "0.6.5+fix", "0.6.01"],
    "0.6.0",
  ],
  ["release/0.6", ["0.6.0", "0.6.1-rc0"], "0.6.1"],
  ["release/2.10", ["2.9.99", "2.10.3"], "2.10.4"],
]) {
  test(`${branch} with tags ${JSON.stringify(tags)} plans ${expected} RCs`, () => {
    assert.equal(
      resolveRcBaseVersion(
        branch,
        tags.map((name) => ({ name })),
      ),
      expected,
    );
  });
}

test("resolves a stable Helm-only release", async () => {
  const plan = await resolveReleasePlan({
    loadSourceWorkflow: async () => sourceWorkflow(),
    context: manualContext({
      "release-type": "stable",
      "release-scope": "helm",
      "source-sha": SHA.toUpperCase(),
      version: "1.2.3",
      "helm-version": "1.2.3-rc.1+build.9",
    }),
    listBranches: async () =>
      assert.fail("stable releases do not discover release branches"),
    compareCommits: async () =>
      assert.fail("stable releases do not compare release branch commits"),
  });

  assert.equal(plan.sourceSha, SHA);
  assert.equal(plan.sourceBranch, "");
  assert.equal(plan.releaseLabel, "1.2.3");
  assert.equal(plan.includeHelm, true);
  assert.deepEqual(plan.wheelIds, []);
  assert.deepEqual(plan.containerIds, []);
});

test("resolves a custom rc release and uses the supplied clock", async () => {
  const plan = await resolveReleasePlan({
    loadSourceWorkflow: async () => sourceWorkflow(),
    context: manualContext({
      "release-type": "rc",
      "release-scope": "custom",
      "wheel-ids": "nemo-helix",
      "container-ids": "nhx-api",
      "include-helm": "false",
      "dry-run": "true",
    }),
    listBranches: async () => BRANCHES,
    now: () => new Date("2026-08-27T12:34:56.789Z"),
  });

  assert.equal(plan.sourceSha, SHA);
  assert.equal(plan.sourceBranch, "release/0.10");
  assert.equal(plan.releaseLabel, "rc-20260827123456");
  assert.deepEqual(plan.wheelIds, ["nemo-helix"]);
  assert.deepEqual(plan.containerIds, ["nhx-api"]);
});

test("rejects duplicate custom artifact IDs", async () => {
  await assert.rejects(
    resolveReleasePlan({
      loadSourceWorkflow: async () => sourceWorkflow(),
      context: manualContext({
        "release-type": "rc",
        "release-scope": "custom",
        "wheel-ids": "nemo-helix,nemo-helix",
      }),
      listBranches: async () => BRANCHES,
    }),
    /wheel-ids contains duplicate entries/,
  );
});

for (const [label, context] of [
  ["scheduled", { eventName: "schedule", sha: "b".repeat(40), payload: {} }],
  ["manual", manualContext({ "release-type": "rc" })],
  ["dry-run", manualContext({ "release-type": "rc", "dry-run": "true" })],
]) {
  test(`${label} RCs select the highest release branch and pin its SHA`, async () => {
    const plan = await resolveReleasePlan({
      loadSourceWorkflow: async () => sourceWorkflow(),
      context,
      listBranches: async () => BRANCHES,
      compareCommits: async () =>
        assert.fail("unpinned RCs use the discovered branch head"),
    });

    assert.equal(plan.sourceBranch, "release/0.10");
    assert.equal(plan.sourceSha, SHA);
    assert.equal(plan.sendNotifications, true);
  });
}

test("compares major versions before minor versions and ignores unrelated branches", async () => {
  const plan = await resolveReleasePlan({
    loadSourceWorkflow: async () => sourceWorkflow(),
    context: manualContext({ "release-type": "rc" }),
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
  test(`pinned RCs retain the release series (dry-run=${dryRun})`, async () => {
    const plan = await resolveReleasePlan({
      loadSourceWorkflow: async () => sourceWorkflow(),
      context: manualContext({
        "release-type": "rc",
        "source-sha": ` ${SHA.toUpperCase()} `,
        "dry-run": dryRun,
        "send-notifications": "false",
      }),
      listBranches: async () => [
        { name: "release/0.7", commit: { sha: SHA } },
        { name: "release/0.6", commit: { sha: "c".repeat(40) } },
      ],
      compareCommits: async (base, head) => {
        assert.equal(base, SHA);
        assert.equal(head, SHA);
        return { status: "identical" };
      },
    });

    assert.equal(plan.sourceSha, SHA);
    assert.equal(plan.sourceBranch, "release/0.7");
    assert.equal(plan.sendNotifications, false);
    assert.equal(
      resolveRcBaseVersion(plan.sourceBranch, [{ name: "0.6.0" }]),
      "0.7.0",
    );
  });
}

test("retains an older pinned SHA within the latest release branch", async () => {
  const olderSha = "d".repeat(40);
  const plan = await resolveReleasePlan({
    loadSourceWorkflow: async () => sourceWorkflow(),
    context: manualContext({
      "release-type": "rc",
      "source-sha": olderSha,
    }),
    listBranches: async () => BRANCHES,
    compareCommits: async (base, head) => {
      assert.equal(base, olderSha);
      assert.equal(head, SHA);
      return { status: "ahead" };
    },
  });
  assert.equal(plan.sourceSha, olderSha);
  assert.equal(plan.sourceBranch, "release/0.10");
});

for (const status of ["behind", "diverged", undefined]) {
  test(`rejects pinned RCs outside the release branch (${status})`, async () => {
    await assert.rejects(
      resolveReleasePlan({
        loadSourceWorkflow: async () => sourceWorkflow(),
        context: manualContext({
          "release-type": "rc",
          "source-sha": SHA,
        }),
        listBranches: async () => BRANCHES,
        compareCommits: async () => ({ status }),
      }),
      /must be an ancestor of release\/0\.10/,
    );
  });
}

test("pinned RCs propagate comparison and branch lookup errors", async () => {
  const error = new Error("GitHub lookup failed");
  for (const failedLookup of ["listBranches", "compareCommits"]) {
    await assert.rejects(
      resolveReleasePlan({
        loadSourceWorkflow: async () => sourceWorkflow(),
        context: manualContext({
          "release-type": "rc",
          "source-sha": SHA,
        }),
        listBranches: async () => BRANCHES,
        compareCommits: async () => ({ status: "identical" }),
        [failedLookup]: async () => {
          throw error;
        },
      }),
      error,
    );
  }
});

test("pinned RCs fail when no release branch exists", async () => {
  await assert.rejects(
    resolveReleasePlan({
      loadSourceWorkflow: async () => sourceWorkflow(),
      context: manualContext({ "release-type": "rc", "source-sha": SHA }),
      listBranches: async () => [],
      compareCommits: async () => assert.fail("no branch to compare"),
    }),
    /No release\/X\.X branch found/,
  );
});

for (const branches of [[], [BRANCHES[2]]]) {
  test(`fails without release branches (${branches.length} total branches)`, async () => {
    await assert.rejects(
      resolveReleasePlan({
        loadSourceWorkflow: async () => sourceWorkflow(),
        context: manualContext({ "release-type": "rc" }),
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
      loadSourceWorkflow: async () => sourceWorkflow(),
      context: manualContext({ "release-type": "rc" }),
      listBranches: async () => {
        throw error;
      },
    }),
    error,
  );
});

for (const releaseType of ["rc", "stable"]) {
  test(`rejects invalid explicit ${releaseType} SHAs before branch discovery`, async () => {
    await assert.rejects(
      resolveReleasePlan({
        loadSourceWorkflow: async () => sourceWorkflow(),
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
      loadSourceWorkflow: async () => sourceWorkflow(),
      context: manualContext({
        "release-type": "stable",
        "release-scope": scope,
        "source-sha": SHA,
        version: "1.2.3",
      }),
      listBranches: async () =>
        assert.fail("stable releases do not discover release branches"),
    });

    assert.deepEqual(plan.wheelIds, ["nemo-helix"]);
  }
});

test("an independent wheel is still selectable by id", async () => {
  const plan = await resolveReleasePlan({
    loadSourceWorkflow: async () => sourceWorkflow(),
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
    loadSourceWorkflow: async () => sourceWorkflow(),
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
    loadSourceWorkflow: async () => sourceWorkflow(),
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
        loadSourceWorkflow: async () => sourceWorkflow(),
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
    { "wheel-ids": "nemo-helix,nemo-sandboxed-gym" },
    { "wheel-ids": "nemo-sandboxed-gym", "container-ids": "nhx-api" },
    { "wheel-ids": "nemo-sandboxed-gym", "include-helm": "true" },
  ];
  for (const extra of selections) {
    await assert.rejects(
      resolveReleasePlan({
        loadSourceWorkflow: async () => sourceWorkflow(),
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
    loadSourceWorkflow: async () => sourceWorkflow(),
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
    loadSourceWorkflow: async () => sourceWorkflow(),
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
    loadSourceWorkflow: async () => sourceWorkflow(),
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

test("an independent wheel cannot be released as an RC", async () => {
  await assert.rejects(
    resolveReleasePlan({
      loadSourceWorkflow: async () => sourceWorkflow(),
      context: manualContext({
        "release-type": "rc",
        "release-scope": "custom",
        "wheel-ids": "nemo-sandboxed-gym",
        "dry-run": "true",
      }),
      listBranches: async () => BRANCHES,
    }),
    /versioned from its own tags/,
  );
});
