// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  dispatchRelease,
  validateDispatchTarget,
} = require("../release-dispatch.cjs");

const SHA = "a".repeat(40);
function harness(eventType, clientPayload) {
  const requests = [];
  const logs = [];
  return {
    requests,
    logs,
    args: {
      eventType,
      clientPayload,
      env: {
        DISPATCH_REPO: "example/builds",
        CI_LEGACY_DISPATCH_REF: "maintenance/legacy",
      },
      core: { info: (message) => logs.push(message) },
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

test("nightly container dispatch preserves the release payload and separates source and tooling refs", async () => {
  const payload = {
    ref: SHA,
    cadence: "nightly",
    version: "nightly-20261002162208",
    containers: ["nmp-api"],
    collect_sources: true,
    bake_env: { NMP_COLLECT_SOURCES: "1" },
  };
  const { args, requests } = harness("release", payload);
  await dispatchRelease(args);
  assert.deepEqual(requests, [
    {
      owner: "example",
      repo: "builds",
      ref: "maintenance/legacy",
      workflow_id: "docker.yaml",
      inputs: { "release-payload": JSON.stringify(payload) },
    },
  ]);
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
      const { args, requests } = harness("stage-wheels", payload);
      await dispatchRelease(args);
      assert.equal(requests[0].workflow_id, "wheels.yaml");
      assert.deepEqual(
        JSON.parse(requests[0].inputs["release-payload"]),
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
    const { args, requests } = harness("register-release-artifacts", payload);
    await dispatchRelease(args);
    assert.deepEqual(requests[0], {
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

test("ACT logs workflow inputs without sending the request", async () => {
  const { args, requests, logs } = harness("release", { ref: SHA });
  args.env.ACT = "true";
  await dispatchRelease(args);
  assert.deepEqual(requests, []);
  assert.equal(
    JSON.parse(logs[0].split("would dispatch: ")[1]).workflow_id,
    "docker.yaml",
  );
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

test("unknown events and API failures are reported without logging success", async () => {
  const { args, logs } = harness("unsupported", {});
  await assert.rejects(dispatchRelease(args), /Unsupported release event/);
  args.eventType = "release";
  args.github.rest.actions.createWorkflowDispatch = async () => {
    throw new Error("API rejected request");
  };
  await assert.rejects(dispatchRelease(args), /API rejected request/);
  assert.deepEqual(logs, []);
});
