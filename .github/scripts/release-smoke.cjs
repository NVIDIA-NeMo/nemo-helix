// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const console = require("node:console");

function resolveRelease(version) {
  const match = /^(\d+\.\d+\.\d+)(?:-(rc-\d{14}|(?:a|b|rc)\d+))?$/.exec(
    version,
  );
  if (!match || match[0] !== version)
    throw new Error(
      "Use a full chart version, e.g. 0.8.0 or 0.8.0-rc-20261021000000",
    );
  const [, base, suffix] = match;
  const rc = suffix?.startsWith("rc-");
  return {
    chart_version: version,
    wheel_version: rc
      ? `${base}.dev${suffix.slice(3)}`
      : `${base}${suffix || ""}`,
    wheel_index: rc ? "https://pypi.nvidia.com" : "https://pypi.org/simple",
    image_registry: rc
      ? "ghcr.io/nvidia-nemo/nemo-helix"
      : "nvcr.io/nvidia/nemo-helix",
    image_tag: rc ? suffix : version,
    chart_repository: rc
      ? "oci://ghcr.io/nvidia-nemo/nemo-helix/nemo-helix"
      : "https://helm.ngc.nvidia.com/nvidia/nemo-helix",
  };
}

function smokeEligible(env) {
  return (
    env.RELEASE_SCOPE === "all" &&
    ["stable", "rc"].includes(env.RELEASE_TYPE) &&
    env.POLL_RESULT === "success" &&
    env.DRY_RUN !== "true" &&
    env.ACT !== "true"
  );
}

function summarize(results) {
  if (results.some((result) => result === "failure")) return "failed";
  if (results.length > 0 && results.every((result) => result === "success"))
    return "passed";
  return "cancelled/incomplete";
}

async function finishSmoke({ env, fetchImpl, github, context }) {
  const result = summarize([
    env.PREPARE_RESULT,
    env.MACOS_RESULT,
    env.KUBERNETES_RESULT,
  ]);
  const icons = {
    passed: ":gitlab_ci_status_success:",
    failed: ":gitlab_ci_status_failed:",
    "cancelled/incomplete": ":gitlab_ci_status_canceled:",
  };
  const failures = [];
  if (result === "failed" && github) {
    try {
      const jobs = await github.paginate(
        github.rest.actions.listJobsForWorkflowRun,
        {
          ...context.repo,
          run_id: context.runId,
          filter: "latest",
        },
      );
      for (const job of jobs) {
        const name = job.name.split(" / ").at(-1);
        if (
          ![
            "prepare",
            "Published wheel (macOS)",
            "Published images and chart",
          ].includes(name)
        )
          continue;
        for (const step of job.steps || []) {
          if (step.conclusion === "failure")
            failures.push(
              `*Failed step:* <${job.html_url}|${name}: ${step.name}>`,
            );
        }
      }
    } catch (error) {
      console.warn(`Could not read failed smoke steps: ${error.message}`);
    }
  }
  const text = [
    `${icons[result]} *Release smoke checks ${result}*`,
    `Release: \`${env.RELEASE_VERSION}\``,
    "",
    `:macos: *macOS:* ${env.MACOS_RESULT}`,
    `:kubernetes: *Kubernetes:* ${env.KUBERNETES_RESULT}`,
    ...failures,
    "",
    `<${env.RUN_URL}|View run and diagnostics>`,
  ].join("\n");
  if (env.SEND_NOTIFICATIONS === "true") {
    if (!env.SLACK_RELEASE_WEBHOOK)
      throw new Error(
        "SLACK_RELEASE_WEBHOOK is required when notifications are enabled",
      );
    const response = await fetchImpl(env.SLACK_RELEASE_WEBHOOK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: globalThis.AbortSignal.timeout(30000),
    });
    if (!response.ok)
      throw new Error(`Slack webhook returned ${response.status}`);
  }
  return { result, text };
}

module.exports = { resolveRelease, smokeEligible, summarize, finishSmoke };
