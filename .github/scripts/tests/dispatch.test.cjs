// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const { dispatchCiConsumer } = require("../dispatch.cjs");

const SHA = "a".repeat(40);
const SOURCE = { owner: "example", repo: "platform" };

function harness({ branch = "release/0.6", pages = [[]], act = "false" } = {}) {
  const requests = [];
  const logs = [];
  const tagReads = [];
  const listTags = () => assert.fail("Tags must be read through pagination.");
  const github = {
    rest: {
      repos: {
        listTags,
        createDispatchEvent: async (request) => requests.push(request),
      },
    },
    paginate: async (method, parameters) => {
      assert.equal(method, listTags);
      assert.deepEqual(parameters, { ...SOURCE, per_page: 100 });
      tagReads.push(parameters);
      return pages.flat().map((name) => ({ name }));
    },
  };
  return {
    requests,
    logs,
    tagReads,
    args: {
      core: { info: (message) => logs.push(message) },
      github,
      context: { ref: `refs/heads/${branch}`, sha: SHA, repo: SOURCE },
      env: { DISPATCH_REPO: "example/builds", ACT: act },
    },
  };
}

test("preserves the main payload without discovering tags", async () => {
  const { args, requests, tagReads } = harness({ branch: "main" });
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

for (const [pages, expected] of [
  [[[]], "0.6.0"],
  [[["0.5.8", "0.7.9", "0.6.0-rc.1", "v0.6.5", "0.6.5+fix"]], "0.6.0"],
  [[["0.6.0"]], "0.6.1"],
  [[["0.6.9", "0.6.2"], ["0.6.10"]], "0.6.11"],
]) {
  test(`dispatches the exact source and next version ${expected} for ${JSON.stringify(pages)}`, async () => {
    const { args, requests, tagReads } = harness({ pages });
    await dispatchCiConsumer(args);
    assert.equal(tagReads.length, 1);
    assert.deepEqual(requests, [
      {
        owner: "example",
        repo: "builds",
        event_type: "release-branch-updated",
        client_payload: {
          ref: SHA,
          branch: "release/0.6",
          version: "0.6",
          release_version: expected,
        },
      },
    ]);
  });
}

for (const branch of ["main", "release/0.6"]) {
  test(`ACT validates and displays ${branch} without dispatching`, async () => {
    const { args, requests, logs } = harness({ branch, act: "true" });
    await dispatchCiConsumer(args);
    assert.deepEqual(requests, []);
    const request = JSON.parse(logs[0].split("would dispatch: ")[1]);
    assert.equal(request.client_payload.ref, SHA);
    assert.equal(request.client_payload.branch, branch);
    assert.equal(
      request.client_payload.release_version,
      branch === "main" ? undefined : "0.6.0",
    );
  });
}

for (const branch of [
  "release/0.6.0",
  "release/0.06",
  "release/latest",
  "feature/test",
]) {
  test(`rejects invalid branch ${branch} before any API call`, async () => {
    const { args, requests, tagReads } = harness({ branch, act: "true" });
    await assert.rejects(
      dispatchCiConsumer(args),
      /main or a release\/X.Y branch/,
    );
    assert.deepEqual(requests, []);
    assert.deepEqual(tagReads, []);
  });
}

test("rejects a tag ref even if its name resembles a release branch", async () => {
  const { args, requests, tagReads } = harness();
  args.context.ref = "refs/tags/release/0.6";
  await assert.rejects(
    dispatchCiConsumer(args),
    /main or a release\/X.Y branch/,
  );
  assert.deepEqual(requests, []);
  assert.deepEqual(tagReads, []);
});

for (const sha of [undefined, "", "abcdef0"]) {
  test(`rejects invalid source SHA ${sha}`, async () => {
    const { args, requests, tagReads } = harness();
    args.context.sha = sha;
    await assert.rejects(dispatchCiConsumer(args), /40-character source SHA/);
    assert.deepEqual(requests, []);
    assert.deepEqual(tagReads, []);
  });
}

test("rejects a missing dispatch destination before any API call", async () => {
  const { args, requests, tagReads } = harness();
  delete args.env.DISPATCH_REPO;
  await assert.rejects(dispatchCiConsumer(args), /DISPATCH_REPO/);
  assert.deepEqual(requests, []);
  assert.deepEqual(tagReads, []);
});

test("rejects an unrepresentable resolved version without dispatching", async () => {
  const { args, requests } = harness({
    branch: `release/${"9".repeat(310)}.6`,
  });
  await assert.rejects(dispatchCiConsumer(args), /Resolved release_version/);
  assert.deepEqual(requests, []);
});

test("tag discovery failure prevents dispatch", async () => {
  const { args, requests } = harness();
  args.github.paginate = async () => {
    throw new Error("Tag lookup failed");
  };
  await assert.rejects(dispatchCiConsumer(args), /Tag lookup failed/);
  assert.deepEqual(requests, []);
});

test("dispatch failure fails the helper without logging success", async () => {
  const { args, logs } = harness();
  args.github.rest.repos.createDispatchEvent = async () => {
    throw new Error("Dispatch failed");
  };
  await assert.rejects(dispatchCiConsumer(args), /Dispatch failed/);
  assert.deepEqual(logs, []);
});
