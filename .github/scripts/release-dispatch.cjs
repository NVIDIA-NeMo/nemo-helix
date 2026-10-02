// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

function validateDispatchTarget(env) {
  const destination = /^([^/\s]+)\/([^/\s]+)$/.exec(env.DISPATCH_REPO ?? "");
  if (!destination) {
    throw new Error("DISPATCH_REPO must be an owner/repository.");
  }
  const ref = env.CI_LEGACY_DISPATCH_REF?.trim();
  if (!ref) {
    throw new Error(
      "CI_LEGACY_DISPATCH_REF must select the maintenance tooling branch.",
    );
  }
  return { owner: destination[1], repo: destination[2], ref };
}

async function dispatchRelease({
  core,
  github,
  env,
  eventType,
  clientPayload,
}) {
  const workflows = {
    "release-branch-updated": "docker.yaml",
    release: "docker.yaml",
    "stage-wheels": "wheels.yaml",
    "register-release-artifacts": "release-register-artifacts.yaml",
  };
  if (!Object.hasOwn(workflows, eventType)) {
    throw new Error(`Unsupported release event: ${eventType}`);
  }
  const payload = clientPayload;
  const inputs =
    eventType === "register-release-artifacts"
      ? {
          environment: "prod",
          version: payload.version,
          "source-sha": payload.source_sha,
          "source-run-url": payload.source_run_url,
          "wheel-ids": payload.wheel_ids.join(","),
          "wheel-version": payload.wheel_version || "",
          "container-ids": payload.container_ids.join(","),
          "container-tag": payload.source_sha,
          "source-team": "nemo-platform-dev",
          "helm-id": payload.helm_id || "",
          "helm-version": payload.helm_version || "",
        }
      : {
          [eventType === "release-branch-updated"
            ? "release-branch-payload"
            : "release-payload"]: JSON.stringify(payload),
        };
  const request = {
    ...validateDispatchTarget(env),
    workflow_id: workflows[eventType],
    inputs,
  };
  if (env.ACT === "true") {
    core.info(`ACT=true; would dispatch: ${JSON.stringify(request)}`);
    return;
  }
  await github.rest.actions.createWorkflowDispatch(request);
  core.info(`Dispatched ${eventType}: ${JSON.stringify(request)}`);
}

module.exports = { dispatchRelease, validateDispatchTarget };
