// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const { resolveNvskillsRequest } = require("../request-nvskills-ci.cjs");

const SKILL =
  "plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/";

function harness({
  role = "maintain",
  actor = "maintainer",
  triggeringActor = actor,
  body = "/nvskills-ci",
  state = "open",
  base = "main",
  fork = false,
  files = [{ filename: `${SKILL}SKILL.md` }],
} = {}) {
  const outputs = {};
  const calls = [];
  const permissionUsers = [];
  const listFiles = () => assert.fail("Use pagination for changed files.");
  return {
    outputs,
    calls,
    permissionUsers,
    args: {
      triggeringActor,
      core: {
        info() {},
        setOutput: (key, value) => (outputs[key] = value),
      },
      context: {
        eventName: "issue_comment",
        repo: { owner: "NVIDIA-NeMo", repo: "nemo-platform" },
        actor,
        runId: 456,
        payload: {
          issue: { number: 123, pull_request: {} },
          comment: { id: 789, body },
        },
      },
      github: {
        rest: {
          repos: {
            async getCollaboratorPermissionLevel({ username }) {
              calls.push("permission");
              permissionUsers.push(username);
              return { data: { role_name: role } };
            },
            async getCommit({ ref }) {
              calls.push(ref);
              return { data: { commit: { message: "Skill update\n\nBody" } } };
            },
          },
          pulls: {
            listFiles,
            async get() {
              calls.push("pull");
              return {
                data: {
                  state,
                  head: {
                    sha: "current-head",
                    repo: {
                      full_name: fork
                        ? "user/fork"
                        : "NVIDIA-NeMo/nemo-platform",
                    },
                  },
                  base: {
                    ref: base,
                    repo: { full_name: "NVIDIA-NeMo/nemo-platform" },
                  },
                },
              };
            },
          },
        },
        async paginate(method, parameters) {
          assert.equal(method, listFiles);
          assert.equal(parameters.per_page, 100);
          calls.push("files");
          return files;
        },
      },
    },
  };
}

test("dispatch inputs use the current PR head and repository, with no executable content", async () => {
  const { args, outputs } = harness();
  await resolveNvskillsRequest(args);
  assert.equal(outputs.should_dispatch, "true");
  assert.deepEqual(JSON.parse(outputs.inputs), {
    source_owner: "NVIDIA-NeMo",
    source_repo: "nemo-platform",
    pr_number: "123",
    request_run_id: "456",
    request_head_sha: "current-head",
    request_base_ref: "main",
    request_commit_title: "Skill update",
    request_comment_id: "789",
    requested_by: "maintainer",
  });
});

for (const role of ["write", "read", "triage", "custom-maintainer"]) {
  test(`rejects ${role} before reading the PR`, async () => {
    const { args, outputs, calls } = harness({ role });
    await assert.rejects(resolveNvskillsRequest(args), /maintain or admin/);
    assert.deepEqual(calls, ["permission"]);
    assert.equal(outputs.should_dispatch, "false");
  });
}

test("a write user cannot rerun a maintainer request against the current PR head", async () => {
  const { args, outputs, calls, permissionUsers } = harness({
    actor: "maintainer",
    triggeringActor: "developer",
    role: "write",
  });
  await assert.rejects(resolveNvskillsRequest(args), /maintain or admin/);
  assert.deepEqual(permissionUsers, ["developer"]);
  assert.deepEqual(calls, ["permission"]);
  assert.equal(outputs.should_dispatch, "false");
  assert.equal(outputs.inputs, undefined);
});

for (const role of ["maintain", "admin"]) {
  test(`an authorized ${role} rerun checks and records its initiator`, async () => {
    const { args, outputs, permissionUsers } = harness({
      actor: "original-maintainer",
      triggeringActor: "rerun-maintainer",
      role,
    });
    await resolveNvskillsRequest(args);
    assert.deepEqual(permissionUsers, ["rerun-maintainer"]);
    assert.equal(outputs.should_dispatch, "true");
    const inputs = JSON.parse(outputs.inputs);
    assert.equal(inputs.requested_by, "rerun-maintainer");
    assert.equal(inputs.request_head_sha, "current-head");
  });
}

for (const identity of [undefined, null, "", " "]) {
  test(`fails closed without a triggering identity: ${JSON.stringify(identity)}`, async () => {
    const { args, outputs, calls } = harness();
    args.triggeringActor = identity;
    await assert.rejects(
      resolveNvskillsRequest(args),
      /triggering actor identity/,
    );
    assert.deepEqual(calls, []);
    assert.equal(outputs.should_dispatch, "false");
    assert.equal(outputs.inputs, undefined);
  });
}

for (const options of [
  { fork: true },
  { state: "closed" },
  { base: "release/example" },
]) {
  test(`rejects unsupported PR: ${JSON.stringify(options)}`, async () => {
    const { args, outputs } = harness(options);
    await assert.rejects(
      resolveNvskillsRequest(args),
      /open same-repository PR into main/,
    );
    assert.equal(outputs.should_dispatch, "false");
  });
}

test("skips unrelated plugin code and other skills", async () => {
  const { args, outputs, calls } = harness({
    files: [
      { filename: "skills/nemo-evals-plugin/SKILL.md" },
      { filename: "plugins/nemo-customizer/src/nemo_customizer/api.py" },
      { filename: `${SKILL.slice(0, -1)}-other/SKILL.md` },
    ],
  });
  await resolveNvskillsRequest(args);
  assert.equal(outputs.should_dispatch, "false");
  assert.ok(!calls.includes("current-head"));
});

test("includes renames out of the skill directory and files on later pages", async () => {
  const files = Array.from({ length: 100 }, () => ({ filename: "README.md" }));
  files.push({
    filename: "archived.md",
    previous_filename: `${SKILL}references/recipes.md`,
  });
  const { args, outputs } = harness({ role: "admin", files });
  await resolveNvskillsRequest(args);
  assert.equal(outputs.should_dispatch, "true");
});

for (const body of ["/nvskills-ci-malicious", "/nvskills-ci extra"]) {
  test(`ignores a command prefix: ${body}`, async () => {
    const { args, outputs, calls } = harness({ body });
    await resolveNvskillsRequest(args);
    assert.equal(outputs.should_dispatch, "false");
    assert.deepEqual(calls, []);
  });
}

test("ignores comments on ordinary issues", async () => {
  const { args, outputs, calls } = harness();
  delete args.context.payload.issue.pull_request;
  await resolveNvskillsRequest(args);
  assert.equal(outputs.should_dispatch, "false");
  assert.deepEqual(calls, []);
});

test("API failures never enable dispatch", async () => {
  const { args, outputs } = harness();
  args.github.paginate = async () => {
    throw new Error("API unavailable");
  };
  await assert.rejects(resolveNvskillsRequest(args), /API unavailable/);
  assert.equal(outputs.should_dispatch, "false");
});
