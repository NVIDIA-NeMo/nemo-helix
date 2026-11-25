// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const CUSTOMIZER_SKILL =
  "plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/";

async function resolveNvskillsRequest({
  core,
  github,
  context,
  triggeringActor,
}) {
  core.setOutput("should_dispatch", "false");
  const { issue, comment } = context.payload;
  if (
    context.eventName !== "issue_comment" ||
    !issue?.pull_request ||
    comment?.body?.trim() !== "/nvskills-ci"
  ) {
    core.info("Expected /nvskills-ci on a pull request.");
    return;
  }

  // Reruns retain context.actor from the original event. Authorize the person
  // starting this attempt instead, with no fallback to the original actor.
  if (typeof triggeringActor !== "string" || !triggeringActor.trim()) {
    throw new Error("NVSkills CI requires the triggering actor identity.");
  }

  const { owner, repo } = context.repo;
  const { data: permission } =
    await github.rest.repos.getCollaboratorPermissionLevel({
      owner,
      repo,
      username: triggeringActor,
    });
  if (!["admin", "maintain"].includes(permission.role_name)) {
    throw new Error(
      "NVSkills CI requests require maintain or admin permission.",
    );
  }

  const { data: pr } = await github.rest.pulls.get({
    owner,
    repo,
    pull_number: issue.number,
  });
  if (
    pr.state !== "open" ||
    pr.head.repo?.full_name !== pr.base.repo.full_name ||
    pr.base.ref !== "main"
  ) {
    throw new Error(
      "NVSkills CI requires an open same-repository PR into main.",
    );
  }

  const files = await github.paginate(github.rest.pulls.listFiles, {
    owner,
    repo,
    pull_number: issue.number,
    per_page: 100,
  });
  if (
    !files.some((file) =>
      [file.filename, file.previous_filename].some((path) =>
        path?.startsWith(CUSTOMIZER_SKILL),
      ),
    )
  ) {
    core.info("Skipped: no changes to the Customizer skill.");
    return;
  }

  const { data: commit } = await github.rest.repos.getCommit({
    owner,
    repo,
    ref: pr.head.sha,
  });
  core.setOutput(
    "inputs",
    JSON.stringify({
      source_owner: owner,
      source_repo: repo,
      pr_number: String(issue.number),
      request_run_id: String(context.runId),
      request_head_sha: pr.head.sha,
      request_base_ref: pr.base.ref,
      request_commit_title: commit.commit.message.split("\n")[0],
      request_comment_id: String(comment.id),
      requested_by: triggeringActor,
    }),
  );
  core.setOutput("should_dispatch", "true");
}

module.exports = { resolveNvskillsRequest };
