// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { setTimeout: delay } = require("node:timers/promises");

async function cleanupPagesDeployments({
  core,
  github,
  context,
  dryRun = true,
  now = Date.now,
  sleep = delay,
}) {
  const deployments = (
    await github.paginate(github.rest.repos.listDeployments, {
      ...context.repo,
      environment: "github-pages",
      per_page: 100,
    })
  )
    .filter((deployment) => deployment.environment === "github-pages")
    .sort((a, b) => b.created_at.localeCompare(a.created_at) || b.id - a.id);
  const cutoff = now() - 7 * 24 * 60 * 60 * 1000;
  const candidates = deployments
    .slice(1)
    .filter((deployment) => Date.parse(deployment.created_at) < cutoff);
  let deleted = 0;
  let skipped = 0;

  core.info(
    `Keeping newest github-pages deployment ${deployments[0]?.id ?? "none"} and the last seven days`,
  );
  for (const deployment of candidates) {
    core.info(
      `${dryRun ? "[dry-run] Would delete" : "Deleting"} github-pages deployment ${deployment.id}`,
    );
    if (dryRun) continue;

    try {
      await github.rest.repos.deleteDeployment({
        ...context.repo,
        deployment_id: deployment.id,
      });
      deleted += 1;
    } catch (error) {
      if (error.status !== 404 && error.status !== 422) throw error;
      skipped += 1;
      core.warning(`Skipped deployment ${deployment.id}: ${error.message}`);
    } finally {
      await sleep(1000);
    }
  }
  const summary = { candidates: candidates.length, deleted, skipped };
  core.info(`Summary: ${JSON.stringify(summary)}`);
  return summary;
}

module.exports = { cleanupPagesDeployments };
