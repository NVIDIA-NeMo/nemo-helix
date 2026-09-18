// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const { dispatchOrReport } = require("../release-dispatch.cjs");

test("ACT reports an act-compatible repository_dispatch envelope", async () => {
  const logs = [];
  const requests = [];
  await dispatchOrReport({
    core: { info: (message) => logs.push(message) },
    github: {
      rest: { repos: { createDispatchEvent: async (request) => requests.push(request) } },
    },
    env: { ACT: "true" },
    eventType: "stage-wheels",
    clientPayload: { ref: "a".repeat(40), wheels: ["nemo-platform"] },
  });

  assert.deepEqual(requests, []);
  assert.deepEqual(
    JSON.parse(logs[0].split("repository_dispatch event:\n")[1]),
    {
      action: "stage-wheels",
      client_payload: { ref: "a".repeat(40), wheels: ["nemo-platform"] },
    },
  );
});

test("GitHub dispatches the event to the configured destination", async () => {
  const requests = [];
  await dispatchOrReport({
    core: { info: () => {} },
    github: {
      rest: { repos: { createDispatchEvent: async (request) => requests.push(request) } },
    },
    env: { DISPATCH_REPO: "NVIDIA-NeMo/Platform-Deploy" },
    eventType: "release",
    clientPayload: { ref: "a".repeat(40), containers: ["nmp-api"] },
  });

  assert.deepEqual(requests, [
    {
      owner: "NVIDIA-NeMo",
      repo: "Platform-Deploy",
      event_type: "release",
      client_payload: { ref: "a".repeat(40), containers: ["nmp-api"] },
    },
  ]);
});
