// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { spawnSync } = require("node:child_process");

const MAX_CONFLICT_FILES = 10;

function escapeSlackText(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\r", "\\r")
    .replaceAll("\n", "\\n");
}

function selectSourcePullRequest({
  pulls,
  forwardPullNumber,
  releaseRef,
  headSha,
}) {
  const candidates = pulls.filter(
    (pull) =>
      pull.number !== forwardPullNumber &&
      pull.merged_at &&
      pull.base?.ref === releaseRef,
  );
  const exact = candidates.filter((pull) => pull.merge_commit_sha === headSha);
  const matches = exact.length > 0 ? exact : candidates;

  return matches.toSorted(
    (left, right) => Date.parse(right.merged_at) - Date.parse(left.merged_at),
  )[0];
}

function parseConflictFiles(output) {
  const lines = output.split(/\r?\n/);
  if (!/^[0-9a-f]{40,64}$/.test(lines[0])) {
    throw new Error("git merge-tree did not return a tree ID");
  }

  const separator = lines.indexOf("", 1);
  if (separator === -1) {
    throw new Error("git merge-tree did not delimit its conflict list");
  }

  return lines.slice(1, separator);
}

function runGit(args, workspace) {
  return spawnSync("git", args, {
    cwd: workspace,
    encoding: "utf8",
    maxBuffer: 10 * 1024 * 1024,
  });
}

function fetchPullTips({ baseSha, headSha, workspace }) {
  const result = runGit(
    ["fetch", "--no-tags", "origin", baseSha, headSha],
    workspace,
  );
  if (result.error || result.status !== 0) {
    throw new Error(
      `git fetch failed: ${result.error?.message || result.stderr.trim()}`,
    );
  }
}

function findConflictFiles({ baseSha, headSha, workspace }) {
  const result = runGit(
    ["merge-tree", "--write-tree", "--name-only", baseSha, headSha],
    workspace,
  );
  if (result.error) {
    throw result.error;
  }
  if (result.status === 0) {
    return { kind: "clean", files: [] };
  }
  if (result.status === 1) {
    return { kind: "conflicts", files: parseConflictFiles(result.stdout) };
  }

  throw new Error(`git merge-tree failed: ${result.stderr.trim()}`);
}

function pullSource(pull) {
  return {
    kind: "pull",
    url: pull.html_url,
    label: `#${pull.number}`,
    authorLogin: pull.user?.login,
  };
}

function commitSource(commit, headSha) {
  return {
    kind: "commit",
    url: commit.html_url,
    label: headSha.slice(0, 7),
    authorLogin: commit.author?.login,
    authorName: commit.commit?.author?.name,
  };
}

async function resolveSource({
  github,
  owner,
  repo,
  forwardPullNumber,
  releaseRef,
  headSha,
}) {
  try {
    const { data: pulls } =
      await github.rest.repos.listPullRequestsAssociatedWithCommit({
        owner,
        repo,
        commit_sha: headSha,
        per_page: 100,
      });
    const pull = selectSourcePullRequest({
      pulls,
      forwardPullNumber,
      releaseRef,
      headSha,
    });
    if (pull) {
      return pullSource(pull);
    }
  } catch {
    // Fall back to commit metadata below.
  }

  const { data: commit } = await github.rest.repos.getCommit({
    owner,
    repo,
    ref: headSha,
  });
  return commitSource(commit, headSha);
}

function sourceLine(source) {
  if (!source) {
    return "Source: unavailable";
  }

  const label = escapeSlackText(source.label);
  let author = source.authorName
    ? escapeSlackText(source.authorName)
    : "unknown author";
  if (source.authorLogin) {
    const login = escapeSlackText(source.authorLogin);
    author = `<https://github.com/${source.authorLogin}|@${login}>`;
  }
  const sourceType = source.kind === "pull" ? "Source PR" : "Source commit";
  return `${sourceType}: <${source.url}|${label}> by ${author}`;
}

function buildSlackMessage({
  repository,
  pullTitle,
  pullUrl,
  commentUrl,
  source,
  conflicts,
  runUrl,
  recoveryDocsUrl,
}) {
  const lines = [
    ":warning: *Forward merge needs attention*",
    `Repository: ${escapeSlackText(repository)}`,
    `PR: <${pullUrl}|${escapeSlackText(pullTitle)}>`,
    sourceLine(source),
  ];

  if (conflicts.kind === "conflicts") {
    lines.push("", `*Conflicting files (${conflicts.files.length}):*`);
    for (const file of conflicts.files.slice(0, MAX_CONFLICT_FILES)) {
      lines.push(`• ${escapeSlackText(file)}`);
    }
    if (conflicts.files.length > MAX_CONFLICT_FILES) {
      const remaining = conflicts.files.length - MAX_CONFLICT_FILES;
      lines.push(`• <${runUrl}|+${remaining} more>`);
    }
  } else if (conflicts.kind === "clean") {
    lines.push(
      "",
      "No Git conflicts detected. The bot failed on another merge requirement.",
    );
  } else {
    lines.push("", "Conflict metadata unavailable; see the failure details.");
  }

  lines.push("", "The bot will not retry. Manual recovery is required.");
  if (recoveryDocsUrl?.trim()) {
    lines.push(
      `:point_right: *<${escapeSlackText(recoveryDocsUrl.trim())}|Follow these steps to fix the forward merge>*`,
    );
  }
  lines.push(`<${commentUrl}|View failure details>`);
  return lines.join("\n");
}

async function writeConflictSummary({ core, pullUrl, files }) {
  const escapedFiles = files.map((file) => escapeSlackText(file));
  await core.summary
    .addHeading("Forward merge conflicts")
    .addLink("Open the forward-merge PR", pullUrl)
    .addList(escapedFiles)
    .write();
}

async function postSlack({ fetchImpl, webhook, text }) {
  const response = await fetchImpl(webhook, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    redirect: "error",
    signal: globalThis.AbortSignal.timeout(30_000),
  });
  if (!response.ok) {
    throw new Error(`Slack webhook returned ${response.status}`);
  }
}

async function sendForwardMergeAlert({
  github,
  context,
  core,
  env,
  fetchImpl,
  workspace,
}) {
  if (!env.SLACK_ALERTS_WEBHOOK) {
    throw new Error("SLACK_ALERTS_WEBHOOK is not set");
  }

  const { owner, repo } = context.repo;
  const issue = context.payload.issue;
  let source;
  let conflicts = { kind: "unavailable", files: [] };

  try {
    const { data: pull } = await github.rest.pulls.get({
      owner,
      repo,
      pull_number: issue.number,
    });

    try {
      source = await resolveSource({
        github,
        owner,
        repo,
        forwardPullNumber: issue.number,
        releaseRef: pull.head.ref,
        headSha: pull.head.sha,
      });
    } catch (error) {
      core.warning(`Unable to resolve source change: ${error.message}`);
    }

    try {
      fetchPullTips({
        baseSha: pull.base.sha,
        headSha: pull.head.sha,
        workspace,
      });
      conflicts = findConflictFiles({
        baseSha: pull.base.sha,
        headSha: pull.head.sha,
        workspace,
      });
    } catch (error) {
      core.warning(`Unable to resolve conflict files: ${error.message}`);
    }
  } catch (error) {
    core.warning(`Unable to read forward-merge PR: ${error.message}`);
  }

  if (
    conflicts.kind === "conflicts" &&
    conflicts.files.length > MAX_CONFLICT_FILES
  ) {
    try {
      await writeConflictSummary({
        core,
        pullUrl: issue.html_url,
        files: conflicts.files,
      });
    } catch (error) {
      core.warning(`Unable to write conflict summary: ${error.message}`);
    }
  }

  const text = buildSlackMessage({
    repository: context.payload.repository.full_name,
    pullTitle: issue.title,
    pullUrl: issue.html_url,
    commentUrl: context.payload.comment.html_url,
    source,
    conflicts,
    runUrl: env.RUN_URL,
    recoveryDocsUrl: env.FORWARD_MERGE_RECOVERY_DOCS_URL,
  });
  await postSlack({
    fetchImpl,
    webhook: env.SLACK_ALERTS_WEBHOOK,
    text,
  });

  return { text, source, conflicts };
}

module.exports = {
  MAX_CONFLICT_FILES,
  buildSlackMessage,
  escapeSlackText,
  findConflictFiles,
  parseConflictFiles,
  postSlack,
  resolveSource,
  selectSourcePullRequest,
  sendForwardMergeAlert,
};
