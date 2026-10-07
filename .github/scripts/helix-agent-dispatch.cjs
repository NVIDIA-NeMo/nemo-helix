// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Trigger-side helper for the Helix agent dispatch workflow.
//
// Parses the trigger token from a PR comment, reacts 👀 to acknowledge, and sends
// a `repository_dispatch` to the worker repo carrying the PR context the agent
// needs. Fire-and-forget: it does not wait for the agent.
//
// The envelope `client_payload` is the documented contract agents conform to:
//   { trigger, repo, pr_number, comment_id, comment_body, comment_author,
//     head_sha, base_ref, head_ref }
// The worker resolves `trigger` + `repo` against dispatch-registry.yaml and
// forwards this payload (plus a minted GitHub token) into the agent job.

const EVENT_TYPE = "helix-agent";

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
    core.info("No /helix-* trigger token at the start of the comment; skipping.");
    return null;
  }

  const [owner, repo] = (context.payload.repository?.full_name ?? "").split("/");
  const clientPayload = {
    trigger,
    repo: context.payload.repository?.full_name,
    pr_number: issue?.number,
    comment_id: comment?.id,
    comment_body: comment?.body,
    comment_author: comment?.user?.login,
    // The PR head SHA pins the review target; filled by the worker from the PR if absent.
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
    core.info(`ACT=true; repository_dispatch event:\n${JSON.stringify(envelope, null, 2)}`);
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
  core.info(`Dispatched ${trigger} for ${clientPayload.repo}#${clientPayload.pr_number}`);
  return envelope;
}

module.exports = { dispatchAgent, parseTrigger, EVENT_TYPE };
