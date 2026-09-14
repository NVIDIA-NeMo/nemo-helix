// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const {
  buildSlackMessage,
  findConflictFiles,
  resolveSource,
  selectSourcePullRequest,
  sendForwardMergeAlert,
} = require("../forward-merge-alert.cjs");

function git(workspace, ...args) {
  return execFileSync("git", ["-c", "core.fsmonitor=false", ...args], {
    cwd: workspace,
    encoding: "utf8",
  }).trim();
}

function commitFile(workspace, filename, contents, message) {
  fs.writeFileSync(path.join(workspace, filename), contents);
  git(workspace, "add", filename);
  git(
    workspace,
    "-c",
    "user.name=Test User",
    "-c",
    "user.email=test@example.com",
    "commit",
    "-m",
    message,
  );
  return git(workspace, "rev-parse", "HEAD");
}

function createDivergedRepository({ conflict }) {
  const workspace = fs.mkdtempSync(
    path.join(os.tmpdir(), "nemo-forward-merge-alert-"),
  );
  git(workspace, "init", "--initial-branch=main");
  const initial = commitFile(workspace, "shared.txt", "initial\n", "initial");

  git(workspace, "checkout", "-b", "base");
  const baseSha = commitFile(workspace, "shared.txt", "base\n", "base");

  git(workspace, "checkout", "-b", "release", initial);
  const filename = conflict ? "shared.txt" : "head.txt";
  const headSha = commitFile(workspace, filename, "head\n", "head");

  return { workspace, baseSha, headSha };
}

function pullFixture(overrides = {}) {
  return {
    number: 1789,
    merged_at: "2026-09-04T15:19:10Z",
    merge_commit_sha: "release-head",
    base: { ref: "release/0.5" },
    user: { login: "soluwalana" },
    html_url: "https://github.com/NVIDIA-NeMo/nemo-platform/pull/1789",
    ...overrides,
  };
}

function messageFixture(overrides = {}) {
  return {
    repository: "NVIDIA-NeMo/nemo-platform",
    pullTitle: "Forward-merge release/0.5 into main",
    pullUrl: "https://github.com/NVIDIA-NeMo/nemo-platform/pull/1803",
    commentUrl:
      "https://github.com/NVIDIA-NeMo/nemo-platform/pull/1803#issuecomment-1",
    source: {
      kind: "pull",
      url: "https://github.com/NVIDIA-NeMo/nemo-platform/pull/1789",
      label: "#1789",
      authorLogin: "soluwalana",
    },
    conflicts: { kind: "conflicts", files: ["uv.lock"] },
    runUrl: "https://github.com/NVIDIA-NeMo/nemo-platform/actions/runs/1234",
    ...overrides,
  };
}

test("selects the merged source PR for the release head", () => {
  const source = selectSourcePullRequest({
    pulls: [
      pullFixture({ number: 1803 }),
      pullFixture(),
      pullFixture({
        number: 1700,
        merged_at: "2026-09-03T10:00:00Z",
        merge_commit_sha: "older",
      }),
    ],
    forwardPullNumber: 1803,
    releaseRef: "release/0.5",
    headSha: "release-head",
  });

  assert.equal(source.number, 1789);
  assert.equal(source.user.login, "soluwalana");
});

test("falls back to release-head commit metadata when no source PR exists", async () => {
  const source = await resolveSource({
    github: {
      rest: {
        repos: {
          listPullRequestsAssociatedWithCommit: async () => ({ data: [] }),
          getCommit: async () => ({
            data: {
              html_url:
                "https://github.com/NVIDIA-NeMo/nemo-platform/commit/abcdef1",
              author: null,
              commit: { author: { name: "Release Author" } },
            },
          }),
        },
      },
    },
    owner: "NVIDIA-NeMo",
    repo: "nemo-platform",
    forwardPullNumber: 1803,
    releaseRef: "release/0.5",
    headSha: "abcdef1234567890",
  });

  assert.deepEqual(source, {
    kind: "commit",
    url: "https://github.com/NVIDIA-NeMo/nemo-platform/commit/abcdef1",
    label: "abcdef1",
    authorLogin: undefined,
    authorName: "Release Author",
  });
});

test("finds conflicts using git merge-tree", (t) => {
  const fixture = createDivergedRepository({ conflict: true });
  t.after(() => fs.rmSync(fixture.workspace, { recursive: true, force: true }));

  assert.deepEqual(findConflictFiles(fixture), {
    kind: "conflicts",
    files: ["shared.txt"],
  });
});

test("distinguishes a clean merge from a conflict", (t) => {
  const fixture = createDivergedRepository({ conflict: false });
  t.after(() => fs.rmSync(fixture.workspace, { recursive: true, force: true }));

  assert.deepEqual(findConflictFiles(fixture), { kind: "clean", files: [] });
});

test("shows only ten conflict files and links the remainder", () => {
  const files = Array.from({ length: 12 }, (_, index) => `file-${index}.txt`);
  const message = buildSlackMessage(
    messageFixture({ conflicts: { kind: "conflicts", files } }),
  );

  assert.match(message, /\*Conflicting files \(12\):\*/);
  assert.match(message, /file-9\.txt/);
  assert.doesNotMatch(message, /file-10\.txt/);
  assert.match(message, /\+2 more/);
});

test("includes the configured recovery guide for every failure kind", () => {
  for (const kind of ["conflicts", "clean", "unavailable"]) {
    const message = buildSlackMessage(
      messageFixture({
        conflicts: { kind, files: ["uv.lock"] },
        recoveryDocsUrl: " https://docs.example.com/forward-merge/ ",
      }),
    );
    assert.match(
      message,
      /<https:\/\/docs\.example\.com\/forward-merge\/\|Follow these steps to fix the forward merge>/,
    );
    assert.match(message, /View failure details/);
  }
});

test("omits the recovery guide when its secret is missing or blank", () => {
  for (const recoveryDocsUrl of [undefined, "", "   "]) {
    const message = buildSlackMessage(messageFixture({ recoveryDocsUrl }));
    assert.doesNotMatch(message, /Follow these steps to fix the forward merge/);
    assert.match(message, /Manual recovery is required/);
  }
});

test("escapes untrusted Slack labels", () => {
  const message = buildSlackMessage(
    messageFixture({
      pullTitle: "Forward <merge> & retry",
      conflicts: { kind: "conflicts", files: ["file<alert>&.txt"] },
    }),
  );

  assert.match(message, /Forward &lt;merge&gt; &amp; retry/);
  assert.match(message, /file&lt;alert&gt;&amp;\.txt/);
});

test("describes a non-conflict merge failure accurately", () => {
  const message = buildSlackMessage(
    messageFixture({ conflicts: { kind: "clean", files: [] } }),
  );

  assert.match(message, /No Git conflicts detected/);
  assert.doesNotMatch(message, /Conflicting files/);
});

test("falls back to the basic alert when PR metadata fails", async () => {
  const warnings = [];
  const requests = [];
  const core = {
    warning: (message) => warnings.push(message),
    summary: {},
  };
  const github = {
    rest: {
      pulls: {
        get: async () => {
          throw new Error("API unavailable");
        },
      },
    },
  };
  const context = {
    repo: { owner: "NVIDIA-NeMo", repo: "nemo-platform" },
    payload: {
      repository: { full_name: "NVIDIA-NeMo/nemo-platform" },
      issue: {
        number: 1803,
        title: "Forward-merge release/0.5 into main",
        html_url: "https://github.com/NVIDIA-NeMo/nemo-platform/pull/1803",
      },
      comment: {
        html_url:
          "https://github.com/NVIDIA-NeMo/nemo-platform/pull/1803#issuecomment-1",
      },
    },
  };

  const result = await sendForwardMergeAlert({
    github,
    context,
    core,
    env: {
      RUN_URL: "https://github.com/NVIDIA-NeMo/nemo-platform/actions/runs/1234",
      SLACK_ALERTS_WEBHOOK: "https://hooks.slack.test/example",
      SLACK_ALERT_USERGROUP_ID: "S0123456789",
      FORWARD_MERGE_RECOVERY_DOCS_URL:
        "https://docs.example.com/forward-merge/",
    },
    fetchImpl: async (url, options) => {
      requests.push({ url, options });
      return { ok: true, status: 200 };
    },
    workspace: ".",
  });

  assert.equal(requests.length, 1);
  assert.equal(requests[0].options.redirect, "error");
  assert.match(
    JSON.parse(requests[0].options.body).text,
    /^:warning: \*Forward merge needs attention\*/,
  );
  assert.doesNotMatch(result.text, /<!subteam\^/);
  assert.match(result.text, /Source: unavailable/);
  assert.ok(
    JSON.parse(requests[0].options.body).text.includes(
      "<https://docs.example.com/forward-merge/|Follow these steps to fix the forward merge>",
    ),
  );
  assert.match(result.text, /Conflict metadata unavailable/);
  assert.match(warnings[0], /Unable to read forward-merge PR/);

  await assert.rejects(
    sendForwardMergeAlert({
      github,
      context,
      core,
      env: {
        RUN_URL:
          "https://github.com/NVIDIA-NeMo/nemo-platform/actions/runs/1234",
      },
      fetchImpl: async () => assert.fail("fetch should not be called"),
      workspace: ".",
    }),
    /SLACK_ALERTS_WEBHOOK is not set/,
  );
});
