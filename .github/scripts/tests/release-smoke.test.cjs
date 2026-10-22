// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const os = require("node:os");
const {
  resolveRelease,
  smokeEligible,
  summarize,
  finishSmoke,
  reportSmoke,
} = require("../release-smoke.cjs");

test("report preserves the failing stage and redacts artifacts even without cleanup", async () => {
  const directory = fs.mkdtempSync(`${os.tmpdir()}/release-smoke-test-`);
  const outputs = {};
  const core = {
    setOutput: (key, value) => {
      outputs[key] = value;
    },
    summary: { addRaw: () => core.summary, write: async () => {} },
  };
  try {
    fs.mkdirSync(`${directory}/logs`);
    fs.writeFileSync(`${directory}/logs/agent.log`, "registry-key github-key");
    fs.writeFileSync(
      `${directory}/result.json`,
      JSON.stringify({
        release: "0.8.0",
        stage: "agent-invoke",
        result: "failed",
      }),
    );
    await reportSmoke({
      core,
      fs,
      env: {
        SMOKE_RESULTS: directory,
        NGC_REGISTRY_READ_TOKEN: "registry-key",
        GH_TOKEN: "github-key",
      },
      steps: { smoke: { outcome: "failure" } },
    });
    assert.equal(outputs.stage, "agent-invoke");
    assert.equal(
      fs.readFileSync(`${directory}/logs/agent.log`, "utf8"),
      "[REDACTED] [REDACTED]",
    );
    fs.unlinkSync(`${directory}/result.json`);
    await reportSmoke({
      core,
      fs,
      env: { SMOKE_RESULTS: directory, RELEASE_VERSION: "0.8.0" },
      steps: { kind: { outcome: "cancelled" } },
    });
    const report = JSON.parse(
      fs.readFileSync(`${directory}/result.json`, "utf8"),
    );
    assert.equal(report.result, "cancelled/incomplete");
    assert.equal(report.stage, "kind");
    assert.equal(report.release, "0.8.0");
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("resolves exact stable, prerelease and nightly artifacts", () => {
  assert.deepEqual(resolveRelease("0.8.0"), {
    chart_version: "0.8.0",
    wheel_version: "0.8.0",
    wheel_index: "https://pypi.org/simple",
    image_registry: "nvcr.io/nvidia/nemo-helix",
    image_tag: "0.8.0",
    chart_repository: "https://helm.ngc.nvidia.com/nvidia/nemo-helix",
  });
  assert.equal(resolveRelease("0.8.0-rc1").wheel_version, "0.8.0rc1");
  const nightly = resolveRelease("0.8.0-nightly-20261021010203");
  assert.equal(nightly.wheel_version, "0.8.0.dev20261021010203");
  assert.equal(nightly.image_tag, "nightly-20261021010203");
  assert.equal(nightly.image_registry, "ghcr.io/nvidia-nemo/nemo-helix");
  assert.equal(nightly.wheel_index, "https://pypi.nvidia.com");
  assert.match(nightly.chart_repository, /^oci:\/\/ghcr.io\//);
  for (const version of [
    "latest",
    "v0.8.0",
    "0.8.0-nightly",
    "0.8.0-nightly-20261021",
    "0.8.0;echo bad",
    "0.8.0\n",
  ]) {
    assert.throws(() => resolveRelease(version));
  }
});

test("only full published releases are eligible, regardless of Slack outcome", () => {
  const env = {
    SMOKE_ENABLED: "true",
    RELEASE_SCOPE: "all",
    RELEASE_TYPE: "stable",
    POLL_RESULT: "success",
  };
  for (const releaseType of ["stable", "nightly"]) {
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
    { SMOKE_ENABLED: "" },
    { RELEASE_SCOPE: "containers" },
    { RELEASE_TYPE: "custom" },
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
    const result = await finishSmoke({
      env: {
        PREPARE_RESULT: "success",
        MACOS_RESULT: macos,
        KUBERNETES_RESULT: "success",
        MACOS_STAGE: "agent-invoke",
        RELEASE_VERSION: "0.8.0",
        SEND_NOTIFICATIONS: "true",
        SLACK_RELEASE_WEBHOOK: "https://example.test/release",
        RUN_URL: "https://example.test/run",
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
    assert.match(calls[0].body.text, /agent-invoke/);
    assert.match(calls[0].body.text, /https:\/\/example.test\/run/);
  }
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
