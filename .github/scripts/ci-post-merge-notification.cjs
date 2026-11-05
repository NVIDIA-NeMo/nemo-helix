// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

function escapeSlack(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function reportPostMergeResults({ core, context, env, fetchImpl }) {
  const results = Object.entries(JSON.parse(env.NEEDS_JSON)).map(
    ([job, value]) => ({ job, result: value.result || "missing" }),
  );
  await core.summary
    .addHeading("Post-merge image and e2e checks")
    .addTable([
      ["Job", "Result"],
      ...results.map(({ job, result }) => [job, result]),
    ])
    .write();

  // Skips are expected for unchanged paths; failures, cancellations, and
  // incomplete results must not silently disappear behind skipped dependents.
  const failures = results.filter(
    ({ result }) => !["success", "skipped"].includes(result),
  );
  if (failures.length === 0) return;

  const description = failures
    .map(({ job, result }) => `${job}: ${result}`)
    .join(", ");
  core.setFailed(`Post-merge image/e2e checks: ${description}`);
  if (!env.SLACK_ALERTS_WEBHOOK)
    throw new Error(
      "SLACK_ALERTS_WEBHOOK is required for post-merge failure alerts",
    );

  const repo = `${context.repo.owner}/${context.repo.repo}`;
  const baseUrl = `${context.serverUrl}/${repo}`;
  const text = [
    ":rotating_light: *Post-merge image/e2e checks failed*",
    `Repository: ${escapeSlack(repo)}`,
    `Commit: <${baseUrl}/commit/${context.sha}|${context.sha.slice(0, 7)}>`,
    ...failures.map(
      ({ job, result }) => `• ${escapeSlack(job)}: ${escapeSlack(result)}`,
    ),
    `<${baseUrl}/actions/runs/${context.runId}|View run and diagnostics>`,
  ].join("\n");
  const response = await fetchImpl(env.SLACK_ALERTS_WEBHOOK, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal: globalThis.AbortSignal.timeout(30000),
  });
  if (!response.ok)
    throw new Error(`Slack webhook returned ${response.status}`);
}

module.exports = { reportPostMergeResults };
