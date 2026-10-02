// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { resolveNightlyBaseVersion } = require("./release-plan.cjs");

async function dispatchCiConsumer({ core, github, context, env }) {
  const destination = /^([^/\s]+)\/([^/\s]+)$/.exec(env.DISPATCH_REPO ?? "");
  if (!destination) {
    throw new Error("DISPATCH_REPO must be an owner/repository.");
  }
  if (!/^[0-9a-f]{40}$/i.test(context.sha ?? "")) {
    throw new Error("Dispatch requires an exact 40-character source SHA.");
  }

  const release = /^refs\/heads\/release\/((0|[1-9]\d*)\.(0|[1-9]\d*))$/.exec(
    context.ref,
  );
  if (context.ref !== "refs/heads/main" && !release) {
    throw new Error("Dispatch requires main or a release/X.Y branch.");
  }

  const request = {
    owner: destination[1],
    repo: destination[2],
    event_type: release ? "release-branch-updated" : "ci-passed",
    client_payload: {
      ref: context.sha,
      branch: context.ref.slice("refs/heads/".length),
      version: release ? release[1] : "",
    },
  };

  if (release) {
    const tags = await github.paginate(github.rest.repos.listTags, {
      ...context.repo,
      per_page: 100,
    });
    // Share nightlies' next-patch rule; resolve against tags at dispatch time.
    const version = resolveNightlyBaseVersion(
      request.client_payload.branch,
      tags,
    );
    if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(version)) {
      throw new Error(
        "Resolved release_version must be an exact X.Y.Z version.",
      );
    }
    request.client_payload.release_version = version;
  }

  if (env.ACT === "true") {
    core.info(`ACT=true; would dispatch: ${JSON.stringify(request)}`);
    return;
  }
  await github.rest.repos.createDispatchEvent(request);
  core.info(
    `Dispatched ${request.event_type}: ${JSON.stringify(request.client_payload)}`,
  );
}

module.exports = { dispatchCiConsumer };
