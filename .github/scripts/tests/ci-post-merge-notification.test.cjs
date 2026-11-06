// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const { reportPostMergeResults } = require("../ci-post-merge-notification.cjs");

function harness(results) {
  const calls = [];
  const failures = [];
  const summary = {
    addHeading() {
      return this;
    },
    addTable(rows) {
      this.rows = rows;
      return this;
    },
    async write() {},
  };
  return {
    calls,
    failures,
    core: { summary, setFailed: (message) => failures.push(message) },
    context: {
      serverUrl: "https://github.com",
      repo: { owner: "NVIDIA-NeMo", repo: "nemo-helix" },
      sha: "a".repeat(40),
      runId: 123,
    },
    env: {
      NEEDS_JSON: JSON.stringify(results),
      SLACK_ALERTS_WEBHOOK: "https://example.test/slack",
    },
    fetchImpl: async (url, options) => {
      calls.push({ url, options, text: JSON.parse(options.body).text });
      return { ok: true };
    },
  };
}

test("successful and path-filtered checks do not require or call Slack", async () => {
  const h = harness({
    build: { result: "success" },
    kind: { result: "skipped" },
  });
  delete h.env.SLACK_ALERTS_WEBHOOK;
  await reportPostMergeResults(h);
  assert.equal(h.calls.length, 0);
  assert.equal(h.failures.length, 0);
  assert.deepEqual(h.core.summary.rows, [
    ["Job", "Result"],
    ["build", "success"],
    ["kind", "skipped"],
  ]);
});

test("Safe Synthesizer failure alerts even when downstream e2e is skipped", async () => {
  const h = harness({
    "build-safe-synthesizer-test-image": { result: "failure" },
    "kind-cpu-e2e": { result: "skipped" },
  });
  await reportPostMergeResults(h);
  assert.equal(h.failures.length, 1);
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].url, h.env.SLACK_ALERTS_WEBHOOK);
  assert.match(h.calls[0].text, /build-safe-synthesizer-test-image: failure/);
  assert.match(h.calls[0].text, /actions\/runs\/123/);
  assert.match(h.calls[0].text, /\/commit\/a{40}/);
  assert.doesNotMatch(h.calls[0].text, /kind-cpu-e2e/);
});

test("prerequisite failures, cancelled tests, and missing outcomes alert", async () => {
  for (const [job, result] of [
    ["policy-wasm", "failure"],
    ["kind-cpu-e2e", "cancelled"],
    ["changes", undefined],
  ]) {
    const h = harness({ [job]: { result }, dependent: { result: "skipped" } });
    await reportPostMergeResults(h);
    assert.equal(h.failures.length, 1);
    assert.equal(h.calls.length, 1);
    assert.ok(h.calls[0].text.includes(`${job}: ${result || "missing"}`));
  }
});

test("missing webhook or failed delivery is visible, not silently ignored", async () => {
  const h = harness({ build: { result: "failure" } });
  delete h.env.SLACK_ALERTS_WEBHOOK;
  await assert.rejects(
    reportPostMergeResults(h),
    /SLACK_ALERTS_WEBHOOK is required/,
  );
  assert.equal(h.calls.length, 0);
  assert.equal(h.failures.length, 1);
  h.env.SLACK_ALERTS_WEBHOOK = "https://example.test/slack";
  h.fetchImpl = async () => ({ ok: false, status: 503 });
  await assert.rejects(reportPostMergeResults(h), /Slack webhook returned 503/);
});

test("job text cannot inject Slack mentions", async () => {
  const h = harness({ "<@here>&": { result: "failure" } });
  await reportPostMergeResults(h);
  assert.match(h.calls[0].text, /&lt;@here&gt;&amp;/);
  assert.doesNotMatch(h.calls[0].text, /<@here>/);
});
