// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { Buffer } = require("node:buffer");
const { execFileSync } = require("node:child_process");
const { readFileSync } = require("node:fs");
const { setTimeout: sleep } = require("node:timers/promises");
const { readReleaseCatalog } = require("./release-plan.cjs");

const RUNBOOK =
  "https://aire.gitlab-master-pages.nvidia.com/platform/runbooks/github/configure-container-package-access/";

function addedImages(base, current) {
  const existing = new Set(
    readReleaseCatalog(base).containers.map(({ id }) => id),
  );
  return readReleaseCatalog(current)
    .containers.map(({ id }) => id)
    .filter((id) => !existing.has(id));
}

async function selectImages({ github, context, core }) {
  const pull = context.payload.pull_request;
  const path = ".github/workflows/release.yaml";
  const { data } = await github.rest.repos.getContent({
    ...context.repo,
    path,
    ref: pull.base.sha,
  });
  const images = addedImages(
    Buffer.from(data.content, "base64").toString("utf8"),
    readFileSync(path, "utf8"),
  );
  if (images.length && pull.head.repo.full_name !== pull.base.repo.full_name) {
    throw new Error(
      "New release images need a same-repository PR to check publishing access. " +
        "Ask a maintainer to move this branch into nemo-helix.",
    );
  }
  core.setOutput("images", JSON.stringify(images));
  core.info(
    images.length
      ? `New release images: ${images.join(", ")}`
      : "No new release images.",
  );
}

async function dispatchBootstrap({ github, context, env }) {
  const destination = /^([^/\s]+)\/([^/\s]+)$/.exec(env.DISPATCH_REPO ?? "");
  if (!destination)
    throw new Error("CI_DISPATCH_REPO must be an owner/repository.");
  await github.rest.repos.createDispatchEvent({
    owner: destination[1],
    repo: destination[2],
    event_type: "release-package-access",
    client_payload: {
      sha: context.payload.pull_request.head.sha,
      images: JSON.parse(env.IMAGE_IDS),
      run_id: String(context.runId),
      run_attempt: env.GITHUB_RUN_ATTEMPT,
    },
  });
}

async function waitForBootstrap({
  github,
  context,
  core,
  env,
  now = Date.now,
  wait = sleep,
}) {
  const statusContext = `release-package-bootstrap/${context.runId}/${env.GITHUB_RUN_ATTEMPT}`;
  const deadline = now() + 10 * 60_000;
  while (now() < deadline) {
    const statuses = await github.paginate(
      github.rest.repos.listCommitStatusesForRef,
      {
        ...context.repo,
        ref: context.payload.pull_request.head.sha,
        per_page: 100,
      },
    );
    // GitHub returns newest statuses first. Each CI attempt has its own context.
    const status = statuses.find((item) => item.context === statusContext);
    if (status?.state === "success") {
      core.info(
        `Platform-Deploy publishing check passed. ${status.target_url}`,
      );
      return;
    }
    if (status?.state === "failure" || status?.state === "error") {
      throw new Error(
        `Platform-Deploy could not publish the test images. ${status.target_url} ${RUNBOOK}`,
      );
    }
    await wait(15_000);
  }
  throw new Error(
    `Timed out waiting for Platform-Deploy status ${statusContext}. Check its workflow runs and rerun this job.`,
  );
}

async function checkPackages({
  github,
  context,
  core,
  env,
  execute = execFileSync,
}) {
  const repository = `${context.repo.owner}/${context.repo.repo}`;
  for (const image of JSON.parse(env.IMAGE_IDS)) {
    const name = `${context.repo.repo.toLowerCase()}/${image}`;
    const reference = `ghcr.io/${context.repo.owner.toLowerCase()}/${name}:bootstrap-test`;
    let metadata;
    try {
      ({ data: metadata } =
        await github.rest.packages.getPackageForOrganization({
          org: context.repo.owner,
          package_type: "container",
          package_name: name,
        }));
    } catch (cause) {
      throw new Error(
        `${reference} is missing or inaccessible to Helix CI. ${RUNBOOK}`,
        { cause },
      );
    }
    if (metadata.visibility !== "public") {
      throw new Error(`${reference} must be public. ${RUNBOOK}`);
    }
    if (
      metadata.repository?.full_name?.toLowerCase() !== repository.toLowerCase()
    ) {
      throw new Error(
        `${reference} must be linked to ${repository}. ${RUNBOOK}`,
      );
    }
    try {
      execute("docker", ["manifest", "inspect", reference], {
        stdio: "pipe",
        timeout: 60_000,
      });
    } catch (cause) {
      throw new Error(`Helix CI cannot read ${reference}. ${RUNBOOK}`, {
        cause,
      });
    }
    core.info(`Verified ${reference}`);
  }
}

module.exports = {
  addedImages,
  selectImages,
  dispatchBootstrap,
  waitForBootstrap,
  checkPackages,
};
