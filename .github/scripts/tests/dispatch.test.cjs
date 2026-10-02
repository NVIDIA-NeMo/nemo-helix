// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const { dispatchCiConsumer } = require("../dispatch.cjs");

const SHA = "a".repeat(40);
function harness(branch = "release/0.5", tags = []) {
  const requests = [];
  const logs = [];
  const tagReads = [];
  const listTags = () => assert.fail("Read all tag pages.");
  return {
    requests,
    logs,
    tagReads,
    args: {
      context: {
        ref: `refs/heads/${branch}`,
        sha: SHA,
        repo: { owner: "example", repo: "source" },
      },
      env: {
        DISPATCH_REPO: "example/builds",
        CI_LEGACY_DISPATCH_REF: "maintenance/legacy",
      },
      core: { info: (message) => logs.push(message) },
      github: {
        rest: {
          repos: {
            listTags,
            createDispatchEvent: async (request) => requests.push(request),
          },
          actions: {
            createWorkflowDispatch: async (request) => requests.push(request),
          },
        },
        paginate: async (method, parameters) => {
          assert.equal(method, listTags);
          assert.deepEqual(parameters, {
            owner: "example",
            repo: "source",
            per_page: 100,
          });
          tagReads.push(parameters);
          return tags.map((name) => ({ name }));
        },
      },
    },
  };
}

test("main retains repository dispatch without a maintenance variable or tag lookup", async () => {
  const { args, requests, tagReads } = harness("main");
  delete args.env.CI_LEGACY_DISPATCH_REF;
  await dispatchCiConsumer(args);
  assert.deepEqual(tagReads, []);
  assert.deepEqual(requests, [
    {
      owner: "example",
      repo: "builds",
      event_type: "ci-passed",
      client_payload: { ref: SHA, branch: "main", version: "" },
    },
  ]);
});

for (const minor of ["0.5", "0.6"]) {
  for (const [tags, patch] of [
    [[], 0],
    [[`${minor}.0`], 1],
    [
      [
        `${minor}.9`,
        `${minor}.10`,
        `${minor}.2`,
        "0.7.19",
        `${minor}.20-rc.1`,
        `v${minor}.30`,
      ],
      11,
    ],
  ]) {
    test(`${minor} push selects the exact source and next patch ${patch}`, async () => {
      const { args, requests, tagReads } = harness(`release/${minor}`, tags);
      await dispatchCiConsumer(args);
      assert.equal(tagReads.length, 1);
      assert.deepEqual(requests, [
        {
          owner: "example",
          repo: "builds",
          ref: "maintenance/legacy",
          workflow_id: "docker.yaml",
          inputs: {
            "release-branch-payload": JSON.stringify({
              ref: SHA,
              branch: `release/${minor}`,
              version: minor,
              release_version: `${minor}.${patch}`,
            }),
          },
        },
      ]);
    });
  }
}

for (const branch of ["main", "release/0.5", "release/0.6"]) {
  test(`ACT reports the actual ${branch} request without sending it`, async () => {
    const { args, requests, logs } = harness(branch);
    args.env.ACT = "true";
    await dispatchCiConsumer(args);
    assert.deepEqual(requests, []);
    const request = JSON.parse(logs[0].split("would dispatch: ")[1]);
    const payload =
      branch === "main"
        ? request.client_payload
        : JSON.parse(request.inputs["release-branch-payload"]);
    assert.equal(payload.ref, SHA);
    assert.equal(payload.branch, branch);
  });
}

for (const branch of [
  "release/0.5.2",
  "release/0.05",
  "release/latest",
  "feature/test",
]) {
  test(`rejects invalid branch ${branch}`, async () => {
    const { args, requests, tagReads } = harness(branch);
    await assert.rejects(
      dispatchCiConsumer(args),
      /main or a release\/X.Y branch/,
    );
    assert.deepEqual(requests, []);
    assert.deepEqual(tagReads, []);
  });
}

test("rejects a tag ref that resembles a release branch", async () => {
  const { args } = harness();
  args.context.ref = "refs/tags/release/0.5";
  await assert.rejects(
    dispatchCiConsumer(args),
    /main or a release\/X.Y branch/,
  );
});

for (const sha of [undefined, "", "abcdef0"]) {
  test(`rejects incomplete SHA ${sha}`, async () => {
    const { args, requests } = harness();
    args.context.sha = sha;
    await assert.rejects(dispatchCiConsumer(args), /40-character source SHA/);
    assert.deepEqual(requests, []);
  });
}

for (const variable of ["DISPATCH_REPO", "CI_LEGACY_DISPATCH_REF"]) {
  test(`missing ${variable} prevents push dispatch`, async () => {
    const { args, requests } = harness();
    delete args.env[variable];
    await assert.rejects(dispatchCiConsumer(args), new RegExp(variable));
    assert.deepEqual(requests, []);
  });
}

test("tag lookup failure prevents dispatch", async () => {
  const { args, requests } = harness();
  args.github.paginate = async () => {
    throw new Error("Tag lookup failed");
  };
  await assert.rejects(dispatchCiConsumer(args), /Tag lookup failed/);
  assert.deepEqual(requests, []);
});

test("unrepresentable patch version prevents dispatch", async () => {
  const { args, requests } = harness(`release/${"9".repeat(310)}.6`);
  await assert.rejects(dispatchCiConsumer(args), /Resolved release_version/);
  assert.deepEqual(requests, []);
});

test("dispatch failure does not log success", async () => {
  const { args, logs } = harness();
  args.github.rest.actions.createWorkflowDispatch = async () => {
    throw new Error("Dispatch failed");
  };
  await assert.rejects(dispatchCiConsumer(args), /Dispatch failed/);
  assert.deepEqual(logs, []);
});
