// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const console = require("node:console");
const test = require("node:test");
const {
  resolveRelease,
  smokeEligible,
  summarize,
  finishSmoke,
} = require("../release-smoke.cjs");

test("resolves exact stable, prerelease and rc artifacts", () => {
  assert.deepEqual(resolveRelease("0.8.0"), {
    chart_version: "0.8.0",
    wheel_version: "0.8.0",
    wheel_index: "https://pypi.org/simple",
    image_registry: "nvcr.io/nvidia/nemo-helix",
    image_tag: "0.8.0",
    chart_repository: "https://helm.ngc.nvidia.com/nvidia/nemo-helix",
  });
  assert.equal(resolveRelease("0.8.0-rc1").wheel_version, "0.8.0rc1");
  const rc = resolveRelease("0.8.0-rc-20261021010203");
  assert.equal(rc.wheel_version, "0.8.0.dev20261021010203");
  assert.equal(rc.image_tag, "rc-20261021010203");
  assert.equal(rc.image_registry, "ghcr.io/nvidia-nemo/nemo-helix");
  assert.equal(rc.wheel_index, "https://pypi.nvidia.com");
  assert.match(rc.chart_repository, /^oci:\/\/ghcr.io\//);
  for (const version of [
    "latest",
    "0.8.0-nightly-20261021010203",
    "v0.8.0",
    "0.8.0-rc",
    "0.8.0-rc-20261021",
    "0.8.0;echo bad",
    "0.8.0\n",
  ]) {
    assert.throws(() => resolveRelease(version));
  }
});

test("full published releases are eligible without an opt-in, regardless of Slack outcome", () => {
  const env = {
    RELEASE_SCOPE: "all",
    RELEASE_TYPE: "stable",
    POLL_RESULT: "success",
  };
  for (const releaseType of ["stable", "rc"]) {
    for (const notification of ["success", "failure", "skipped"]) {
      assert.equal(
        smokeEligible({
          ...env,
          RELEASE_TYPE: releaseType,
          NOTIFY_RESULT: notification,
        }),
        true,
      );
    }
  }
  for (const change of [
    { RELEASE_SCOPE: "containers" },
    { RELEASE_TYPE: "custom" },
    { RELEASE_TYPE: "nightly" },
    { POLL_RESULT: "failure" },
    { POLL_RESULT: "skipped" },
    { DRY_RUN: "true" },
    { ACT: "true" },
  ]) {
    assert.equal(smokeEligible({ ...env, ...change }), false);
  }
});

test("success requires every job; failures and cancellations never pass", () => {
  assert.equal(summarize(["success", "success", "success"]), "passed");
  assert.equal(summarize(["success", "failure", "success"]), "failed");
  assert.equal(summarize(["failure", "skipped", "skipped"]), "failed");
  for (const result of ["cancelled", "skipped", "", undefined]) {
    assert.equal(
      summarize(["success", result, "success"]),
      "cancelled/incomplete",
    );
  }
});

test("disabled notifications never call Slack, even on failure", async () => {
  const result = await finishSmoke({
    env: { SEND_NOTIFICATIONS: "false", MACOS_RESULT: "failure" },
    fetchImpl: () => assert.fail("Slack must not be called"),
  });
  assert.equal(result.result, "failed");
});

test("one combined release-channel notification for each terminal outcome", async () => {
  for (const [macos, expected] of [
    ["success", "passed"],
    ["failure", "failed"],
    ["cancelled", "cancelled/incomplete"],
  ]) {
    const calls = [];
    let lookups = 0;
    const result = await finishSmoke({
      env: {
        PREPARE_RESULT: "success",
        MACOS_RESULT: macos,
        KUBERNETES_RESULT: "success",
        RELEASE_VERSION: "0.8.0",
        SEND_NOTIFICATIONS: "true",
        SLACK_RELEASE_WEBHOOK: "https://example.test/release",
        RUN_URL: "https://example.test/run",
      },
      context: { repo: { owner: "test", repo: "helix" }, runId: 123 },
      github: {
        rest: { actions: { listJobsForWorkflowRun: "list-jobs" } },
        paginate: async (method, options) => {
          lookups++;
          assert.equal(method, "list-jobs");
          assert.equal(options.run_id, 123);
          assert.equal(options.filter, "latest");
          return [
            {
              name: "Post-release smoke checks / Published wheel (macOS)",
              html_url: "https://example.test/job/1",
              steps: [
                {
                  name: "Install published package and launch CLI",
                  conclusion: macos,
                },
              ],
            },
            {
              name: "Notify release result",
              steps: [{ name: "Send Slack alert", conclusion: "failure" }],
            },
          ];
        },
      },
      fetchImpl: async (url, request) => {
        calls.push({ url, body: JSON.parse(request.body) });
        return { ok: true };
      },
    });
    assert.equal(result.result, expected);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "https://example.test/release");
    assert.match(calls[0].body.text, /0\.8\.0/);
    assert.match(calls[0].body.text, /https:\/\/example.test\/run/);
    assert.equal(lookups, expected === "failed" ? 1 : 0);
    if (expected === "failed")
      assert.match(
        calls[0].body.text,
        /Failed step:\* <https:\/\/example.test\/job\/1\|Published wheel \(macOS\): Install published package and launch CLI>/,
      );
    else assert.doesNotMatch(calls[0].body.text, /Failed step:/);
    assert.doesNotMatch(calls[0].body.text, /Send Slack alert/);
  }
});

test("job lookup failure still sends the failure alert", async (t) => {
  t.mock.method(console, "warn", () => {});
  let sent = false;
  const { result } = await finishSmoke({
    env: {
      KUBERNETES_RESULT: "failure",
      SEND_NOTIFICATIONS: "true",
      SLACK_RELEASE_WEBHOOK: "https://example.test/release",
    },
    context: { repo: {}, runId: 123 },
    github: {
      rest: { actions: {} },
      paginate: async () => {
        throw new Error("GitHub unavailable");
      },
    },
    fetchImpl: async () => {
      sent = true;
      return { ok: true };
    },
  });
  assert.equal(result, "failed");
  assert.equal(sent, true);
});

test("Slack delivery failure fails the result job", async () => {
  await assert.rejects(
    finishSmoke({
      env: {
        SEND_NOTIFICATIONS: "true",
        SLACK_RELEASE_WEBHOOK: "https://example.test",
      },
      fetchImpl: async () => ({ ok: false, status: 500 }),
    }),
    /500/,
  );
});
