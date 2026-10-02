// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const { mkdtempSync, rmSync, writeFileSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join } = require("node:path");
const test = require("node:test");
const {
  dispatchRelease,
  validateDispatchTarget,
} = require("../release-dispatch.cjs");

const SHA = "a".repeat(40);
const fakeGhDirectory = mkdtempSync(join(tmpdir(), "manual-release-test-"));
writeFileSync(
  join(fakeGhDirectory, "gh"),
  "#!/bin/sh\nprintf '%s\\0' \"$@\"\n",
  {
    mode: 0o755,
  },
);
test.after(() => rmSync(fakeGhDirectory, { recursive: true, force: true }));

function runCommand(command, repository = "example/builds") {
  return spawnSync(
    "/bin/sh",
    ["-c", command.replace("OWNER/REPO", repository)],
    {
      encoding: "utf8",
      env: { PATH: fakeGhDirectory },
    },
  );
}

function readRequest(command) {
  const result = runCommand(command);
  assert.equal(result.status, 0, result.stderr);
  const args = result.stdout.split("\0").slice(0, -1);
  assert.deepEqual(args.slice(0, 2), ["workflow", "run"]);
  assert.equal(args[3], "--repo");
  assert.equal(args[5], "--ref");
  const [owner, repo] = args[4].split("/");
  const inputs = {};
  for (let i = 7; i < args.length; i += 2) {
    assert.equal(args[i], "--raw-field");
    const separator = args[i + 1].indexOf("=");
    inputs[args[i + 1].slice(0, separator)] = args[i + 1].slice(separator + 1);
  }
  return { owner, repo, workflow_id: args[2], ref: args[6], inputs };
}

function harness(eventType, clientPayload) {
  const requests = [];
  const logs = [];
  const summaries = [];
  return {
    requests,
    logs,
    summaries,
    get command() {
      assert.deepEqual(
        requests,
        [],
        "Printing must never dispatch a workflow.",
      );
      assert.equal(logs.length, 1);
      const [heading, ...lines] = logs[0].split("\n");
      assert.equal(
        heading,
        "MANUAL ACTION REQUIRED — RUN THIS COMMAND LOCALLY",
      );
      const command = lines.join("\n");
      assert.deepEqual(summaries, [
        heading,
        { command, language: "bash" },
        "written",
      ]);
      return command;
    },
    get request() {
      return readRequest(this.command);
    },
    args: {
      eventType,
      clientPayload,
      env: {
        DISPATCH_REPO: "example/builds",
        CI_LEGACY_DISPATCH_REF: "maintenance/legacy",
      },
      core: {
        info: (message) => logs.push(message),
        summary: {
          addHeading(heading) {
            summaries.push(heading);
            return this;
          },
          addCodeBlock(command, language) {
            summaries.push({ command, language });
            return this;
          },
          async write() {
            summaries.push("written");
          },
        },
      },
      github: {
        rest: {
          actions: {
            createWorkflowDispatch: async (request) => requests.push(request),
          },
        },
      },
    },
  };
}

test("nightly container command preserves the release payload and separates source and tooling refs", async () => {
  const payload = {
    ref: SHA,
    cadence: "nightly",
    version: "nightly-20261002162208",
    containers: ["nmp-api"],
    collect_sources: true,
    bake_env: { NMP_COLLECT_SOURCES: "1" },
  };
  const h = harness("release", payload);
  await dispatchRelease(h.args);
  assert.deepEqual(h.request, {
    owner: "example",
    repo: "builds",
    ref: "maintenance/legacy",
    workflow_id: "docker.yaml",
    inputs: { "release-payload": JSON.stringify(payload) },
  });
});

for (const cadence of ["nightly", "release"]) {
  for (const publish of [false, true]) {
    test(`wheel ${cadence} request preserves publication=${publish} and exact version`, async () => {
      const payload = {
        ref: SHA,
        cadence,
        release_label: "0.5.2",
        nightly_timestamp: "20261002162208",
        wheel_version:
          cadence === "release" ? "0.5.2" : "0.5.1.dev20261002162208",
        wheels: ["nemo-platform", "nemo-platform-plugin"],
        publish_nightly_wheels: publish,
      };
      const h = harness("stage-wheels", payload);
      await dispatchRelease(h.args);
      assert.equal(h.request.workflow_id, "wheels.yaml");
      assert.deepEqual(
        JSON.parse(h.request.inputs["release-payload"]),
        payload,
      );
    });
  }
}

for (const [wheels, containers, helm] of [
  [
    ["nemo-platform", "nemo-platform-plugin"],
    ["nmp-api", "nmp-gym-tasks"],
    true,
  ],
  [[], ["nmp-api"], false],
  [["nemo-platform-plugin"], [], false],
  [[], [], true],
]) {
  test(`stable registration selects SHA-tagged containers and selected artifacts ${JSON.stringify([wheels, containers, helm])}`, async () => {
    const payload = {
      version: "0.5.2",
      source_sha: SHA,
      source_run_url: "https://github.com/example/source/actions/runs/1",
      wheel_ids: wheels,
      wheel_version: "0.5.2",
      container_ids: containers,
      helm_id: helm ? "nemo-platform" : null,
      helm_version: helm ? "0.5.2" : null,
    };
    const h = harness("register-release-artifacts", payload);
    await dispatchRelease(h.args);
    assert.deepEqual(h.request, {
      owner: "example",
      repo: "builds",
      ref: "maintenance/legacy",
      workflow_id: "release-register-artifacts.yaml",
      inputs: {
        environment: "prod",
        version: "0.5.2",
        "source-sha": SHA,
        "source-run-url": payload.source_run_url,
        "wheel-ids": wheels.join(","),
        "wheel-version": "0.5.2",
        "container-ids": containers.join(","),
        "container-tag": SHA,
        "source-team": "nemo-platform-dev",
        "helm-id": helm ? "nemo-platform" : "",
        "helm-version": helm ? "0.5.2" : "",
      },
    });
  });
}

test("ACT prints the same manual command without sending the request", async () => {
  const live = harness("release", { ref: SHA });
  const local = harness("release", { ref: SHA });
  local.args.env.ACT = "true";
  await dispatchRelease(live.args);
  await dispatchRelease(local.args);
  assert.equal(local.command, live.command);
});

for (const value of [undefined, "", "   "]) {
  test(`rejects empty maintenance ref ${JSON.stringify(value)}`, async () => {
    const { args, requests } = harness("release", { ref: SHA });
    args.env.CI_LEGACY_DISPATCH_REF = value;
    await assert.rejects(dispatchRelease(args), /CI_LEGACY_DISPATCH_REF/);
    assert.deepEqual(requests, []);
  });
}

for (const value of [undefined, "", "example", "example/repo/extra"]) {
  test(`rejects invalid destination ${JSON.stringify(value)}`, () => {
    assert.throws(
      () =>
        validateDispatchTarget({
          DISPATCH_REPO: value,
          CI_LEGACY_DISPATCH_REF: "maintenance/legacy",
        }),
      /DISPATCH_REPO/,
    );
  });
}

test("unknown events fail without printing a command", async () => {
  const { args, logs, summaries } = harness("unsupported", {});
  await assert.rejects(dispatchRelease(args), /Unsupported release event/);
  assert.deepEqual(logs, []);
  assert.deepEqual(summaries, []);
});

test("shell quoting preserves quotes, substitutions, backticks, newlines and equals signs", async () => {
  const value =
    "'\" $(printf injected) `printf injected` \\ $HOME\nsecond=line";
  const payload = { ref: SHA, bake_env: { VALUE: value } };
  const h = harness("release", payload);
  h.args.env.CI_LEGACY_DISPATCH_REF =
    "maintenance/o'hare-$(printf injected)-`printf injected`";
  await dispatchRelease(h.args);
  assert.equal(h.request.ref, h.args.env.CI_LEGACY_DISPATCH_REF);
  assert.deepEqual(JSON.parse(h.request.inputs["release-payload"]), payload);
});

test("the operator fills the repository placeholder; logs and summaries contain no secret values or names", async () => {
  const h = harness("release", { ref: SHA });
  h.args.env.DISPATCH_REPO = "private-owner/private-tooling";
  h.args.env.CI_DISPATCH_TOKEN = "secret-token-sentinel";
  h.args.env.GITHUB_TOKEN = "workflow-token-sentinel";
  await dispatchRelease(h.args);
  const result = runCommand(h.command, "operator/tooling");
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.split("\0")[4], "operator/tooling");
  const printed = JSON.stringify([h.logs, h.summaries]);
  for (const secret of [
    h.args.env.DISPATCH_REPO,
    h.args.env.CI_DISPATCH_TOKEN,
    h.args.env.GITHUB_TOKEN,
    "CI_DISPATCH_REPO",
    "CI_DISPATCH_TOKEN",
    "GITHUB_TOKEN",
    "CI_LEGACY_DISPATCH_REF",
  ]) {
    assert.ok(!printed.includes(secret));
  }
});

test("commands show a plain repository placeholder for the operator to replace", async () => {
  const h = harness("release", { ref: SHA });
  await dispatchRelease(h.args);
  assert.match(h.command, /--repo OWNER\/REPO/);
});
