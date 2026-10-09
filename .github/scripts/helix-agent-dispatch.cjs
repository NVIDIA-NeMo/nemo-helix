// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Trigger-side helper for the Helix agent dispatch workflow: parse the `/helix-*`
// token from a PR comment, react 👀, and fire a `repository_dispatch` to the worker
// (fire-and-forget). Envelope contract + architecture: see HELIX_AGENT_DISPATCH.md.

const EVENT_TYPE = "helix-agent";
// Bound the forwarded comment: repository_dispatch caps client_payload at 64KB.
const MAX_COMMENT_BODY = 10000;

function parseTrigger(body) {
  // First whitespace-delimited token, if it looks like `/helix-<name>`.
  const token = (body ?? "").trim().split(/\s+/, 1)[0] ?? "";
  return /^\/helix-[a-z0-9-]+$/.test(token) ? token : null;
}

async function dispatchAgent({ core, github, context, env }) {
  const comment = context.payload.comment;
  const issue = context.payload.issue;
  const trigger = parseTrigger(comment?.body);
  if (!trigger) {
    core.info(
      "No /helix-* trigger token at the start of the comment; skipping.",
    );
    return null;
  }

  const [owner, repo] = (context.payload.repository?.full_name ?? "").split(
    "/",
  );
  const clientPayload = {
    trigger,
    repo: context.payload.repository?.full_name,
    pr_number: issue?.number,
    comment_id: comment?.id,
    comment_body: (comment?.body ?? "").slice(0, MAX_COMMENT_BODY),
    comment_author: comment?.user?.login,
    // Revision fields are worker-populated (it resolves them from the PR); the
    // producer always emits null here.
    head_sha: null,
    base_ref: null,
    head_ref: null,
  };

  // Acknowledge immediately so the human sees the request was received.
  if (env.ACT !== "true") {
    try {
      await github.rest.reactions.createForIssueComment({
        owner,
        repo,
        comment_id: comment.id,
        content: "eyes",
      });
    } catch (error) {
      core.warning(`Could not add 👀 reaction: ${error.message}`);
    }
  }

  const envelope = { action: EVENT_TYPE, client_payload: clientPayload };
  if (env.ACT === "true") {
    core.info(
      `ACT=true; repository_dispatch event:\n${JSON.stringify(envelope, null, 2)}`,
    );
    return envelope;
  }

  const destination = /^([^/\s]+)\/([^/\s]+)$/.exec(env.DISPATCH_REPO ?? "");
  if (!destination) {
    throw new Error("DISPATCH_REPO must be an owner/repository.");
  }
  await github.rest.repos.createDispatchEvent({
    owner: destination[1],
    repo: destination[2],
    event_type: EVENT_TYPE,
    client_payload: clientPayload,
  });
  core.info(
    `Dispatched ${trigger} for ${clientPayload.repo}#${clientPayload.pr_number}`,
  );
  return envelope;
}

module.exports = { dispatchAgent, parseTrigger, EVENT_TYPE };
