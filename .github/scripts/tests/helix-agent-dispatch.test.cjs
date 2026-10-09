// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  dispatchAgent,
  parseTrigger,
  EVENT_TYPE,
} = require("../helix-agent-dispatch.cjs");

function fakeContext(body) {
  return {
    payload: {
      comment: { id: 42, body, user: { login: "benmccown" } },
      issue: { number: 2653, pull_request: {} },
      repository: { full_name: "NVIDIA-NeMo/nemo-helix" },
    },
  };
}

test("parseTrigger accepts a leading /helix-* token only", () => {
  assert.equal(parseTrigger("/helix-review please"), "/helix-review");
  assert.equal(parseTrigger("/helix-docs"), "/helix-docs");
  assert.equal(parseTrigger("looks good"), null);
  assert.equal(parseTrigger("hey /helix-review"), null);
  assert.equal(parseTrigger(""), null);
});

test("no trigger token -> no reaction, no dispatch", async () => {
  const calls = [];
  const out = await dispatchAgent({
    core: { info: () => {}, warning: () => {} },
    github: {
      rest: {
        reactions: {
          createForIssueComment: async (r) => calls.push(["react", r]),
        },
        repos: {
          createDispatchEvent: async (r) => calls.push(["dispatch", r]),
        },
      },
    },
    context: fakeContext("just a normal comment"),
    env: { DISPATCH_REPO: "NVIDIA-NeMo/Platform-Deploy" },
  });
  assert.equal(out, null);
  assert.deepEqual(calls, []);
});

test("ACT reports the envelope and does not dispatch or react", async () => {
  const logs = [];
  const calls = [];
  const out = await dispatchAgent({
    core: { info: (m) => logs.push(m), warning: () => {} },
    github: {
      rest: {
        reactions: {
          createForIssueComment: async (r) => calls.push(["react", r]),
        },
        repos: {
          createDispatchEvent: async (r) => calls.push(["dispatch", r]),
        },
      },
    },
    context: fakeContext("/helix-review go"),
    env: { ACT: "true", DISPATCH_REPO: "NVIDIA-NeMo/Platform-Deploy" },
  });
  assert.deepEqual(calls, []);
  assert.equal(out.action, EVENT_TYPE);
  assert.equal(out.client_payload.trigger, "/helix-review");
  assert.equal(out.client_payload.pr_number, 2653);
  assert.equal(out.client_payload.comment_author, "benmccown");
});

test("a /helix-* comment reacts 👀 and dispatches with PR context", async () => {
  const calls = [];
  await dispatchAgent({
    core: { info: () => {}, warning: () => {} },
    github: {
      rest: {
        reactions: {
          createForIssueComment: async (r) => calls.push(["react", r]),
        },
        repos: {
          createDispatchEvent: async (r) => calls.push(["dispatch", r]),
        },
      },
    },
    context: fakeContext("/helix-review"),
    env: { DISPATCH_REPO: "NVIDIA-NeMo/Platform-Deploy" },
  });

  const react = calls.find((c) => c[0] === "react")[1];
  assert.deepEqual(react, {
    owner: "NVIDIA-NeMo",
    repo: "nemo-helix",
    comment_id: 42,
    content: "eyes",
  });

  const dispatch = calls.find((c) => c[0] === "dispatch")[1];
  assert.equal(dispatch.owner, "NVIDIA-NeMo");
  assert.equal(dispatch.repo, "Platform-Deploy");
  assert.equal(dispatch.event_type, EVENT_TYPE);
  assert.equal(dispatch.client_payload.trigger, "/helix-review");
  assert.equal(dispatch.client_payload.repo, "NVIDIA-NeMo/nemo-helix");
  assert.equal(dispatch.client_payload.pr_number, 2653);
});
