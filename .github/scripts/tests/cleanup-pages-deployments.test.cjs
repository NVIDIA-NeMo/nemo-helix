// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const { cleanupPagesDeployments } = require("../cleanup-pages-deployments.cjs");

const NOW = Date.parse("2026-10-23T12:00:00Z");
const DAY = 24 * 60 * 60 * 1000;
const REPO = { owner: "NVIDIA-NeMo", repo: "nemo-helix" };
const deployment = (id, daysOld, environment = "github-pages") => ({
  id,
  environment,
  created_at: new Date(NOW - daysOld * DAY).toISOString(),
});

function harness(pages = [], errors = {}) {
  const events = [];
  const logs = [];
  const warnings = [];
  const listDeployments = () => assert.fail("Pagination is required");
  const args = {
    core: {
      info: (message) => logs.push(message),
      warning: (message) => warnings.push(message),
    },
    github: {
      rest: {
        repos: {
          listDeployments,
          deleteDeployment: async ({ deployment_id, ...repo }) => {
            assert.deepEqual(repo, REPO);
            events.push(["delete", deployment_id]);
            if (errors[deployment_id]) {
              throw Object.assign(new Error("GitHub refused deletion"), {
                status: errors[deployment_id],
              });
            }
          },
        },
      },
      paginate: async (method, parameters) => {
        assert.equal(method, listDeployments);
        assert.deepEqual(parameters, {
          ...REPO,
          environment: "github-pages",
          per_page: 100,
        });
        return pages.flat();
      },
    },
    context: { repo: REPO },
    dryRun: false,
    now: () => NOW,
    sleep: async (ms) => events.push(["sleep", ms]),
  };
  return { args, events, logs, warnings };
}

test("paginates, protects the newest even when old, and only deletes Pages records", async () => {
  const { args, events } = harness([
    [deployment(1, 20), deployment(4, 0, "qualified")],
    [deployment(3, 8), deployment(2, 15)],
  ]);
  assert.deepEqual(await cleanupPagesDeployments(args), {
    candidates: 2,
    deleted: 2,
    skipped: 0,
  });
  assert.deepEqual(events, [
    ["delete", 2],
    ["sleep", 1000],
    ["delete", 1],
    ["sleep", 1000],
  ]);
});

test("keeps the last seven days, including the exact cutoff", async () => {
  const { args, events } = harness([
    [deployment(1, 8), deployment(2, 7), deployment(3, 6), deployment(4, 0)],
  ]);
  await cleanupPagesDeployments(args);
  assert.deepEqual(events, [
    ["delete", 1],
    ["sleep", 1000],
  ]);
});

for (const records of [[], [deployment(1, 100)]]) {
  test(`does nothing with ${records.length} deployment(s)`, async () => {
    const { args, events } = harness([records]);
    assert.deepEqual(await cleanupPagesDeployments(args), {
      candidates: 0,
      deleted: 0,
      skipped: 0,
    });
    assert.deepEqual(events, []);
  });
}

test("defaults to dry-run and logs candidates without deleting or sleeping", async () => {
  const { args, events, logs } = harness([
    [deployment(1, 20), deployment(2, 10)],
  ]);
  delete args.dryRun;
  assert.deepEqual(await cleanupPagesDeployments(args), {
    candidates: 1,
    deleted: 0,
    skipped: 0,
  });
  assert.deepEqual(events, []);
  assert.ok(logs.includes("[dry-run] Would delete github-pages deployment 1"));
});

test("skips deletion refusals and already-deleted records, then continues", async () => {
  const { args, events, warnings } = harness(
    [[deployment(4, 0), deployment(3, 8), deployment(2, 9), deployment(1, 10)]],
    { 3: 422, 2: 404 },
  );
  assert.deepEqual(await cleanupPagesDeployments(args), {
    candidates: 3,
    deleted: 1,
    skipped: 2,
  });
  assert.deepEqual(
    events.filter(([event]) => event === "delete"),
    [
      ["delete", 3],
      ["delete", 2],
      ["delete", 1],
    ],
  );
  assert.equal(warnings.length, 2);
});

for (const status of [403, 429, 500]) {
  test(`fails immediately on HTTP ${status}`, async () => {
    const { args, events } = harness(
      [[deployment(3, 0), deployment(2, 8), deployment(1, 9)]],
      { 2: status },
    );
    await assert.rejects(cleanupPagesDeployments(args), { status });
    assert.deepEqual(events, [
      ["delete", 2],
      ["sleep", 1000],
    ]);
  });
}
