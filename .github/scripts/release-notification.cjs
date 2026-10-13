// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

async function sendReleaseNotification({ env, fetchImpl }) {
  const releaseType = env.RELEASE_TYPE;
  const wheelIds = JSON.parse(env.WHEEL_IDS);
  const wheelCatalog = JSON.parse(env.WHEEL_CATALOG);
  const containerIds = JSON.parse(env.CONTAINER_IDS);
  const stagesNightlyWheels =
    releaseType === "nightly" &&
    env.PUBLISH_NIGHTLY_WHEELS !== "true" &&
    wheelIds.length > 0;
  const hasPublishedArtifacts =
    (wheelIds.length > 0 && !stagesNightlyWheels) ||
    containerIds.length > 0 ||
    env.INCLUDE_HELM === "true";
  const results = [
    env.POLL_RESULT,
    env.GITHUB_RELEASE_RESULT,
    env.DEPLOYMENT_RESULT,
  ];
  const failed = results.some((result) =>
    ["failure", "cancelled"].includes(result),
  );
  const published = env.POLL_RESULT === "success" && !failed;
  const webhook = published
    ? env.SLACK_RELEASE_WEBHOOK
    : env.SLACK_ALERTS_WEBHOOK;
  const title = published
    ? releaseType === "stable"
      ? "*:ship: Release publish complete*"
      : "*:crescent_moon: Nightly release complete*"
    : releaseType === "stable"
      ? "*:alert: Release publish failed*"
      : "*:alert: Nightly release publish failed*";
  const lines = [
    title,
    `Release: ${env.RELEASE_LABEL}`,
    `Commit: <${env.COMMIT_URL}|${env.SOURCE_SHA.slice(0, 7)}>`,
  ];

  if (published && releaseType === "nightly") {
    const nightlyInstanceUrl = env.NIGHTLY_INSTANCE_URL?.trim();
    if (!nightlyInstanceUrl) {
      throw new Error(
        "NIGHTLY_INSTANCE_URL secret is required for nightly notifications",
      );
    }
    lines.push(`Nightly instance: <${nightlyInstanceUrl}>`);
  }

  if (published) {
    if (hasPublishedArtifacts) {
      lines.push("", "*Artifacts published:*");
    }
    if (wheelIds.length > 0 && !stagesNightlyWheels) {
      lines.push("*:python: Wheels published:*");
      for (const wheelId of wheelIds) {
        const wheel = wheelCatalog.find(
          (candidate) => candidate.id === wheelId,
        );
        const wheelIndex =
          releaseType === "nightly"
            ? env.NIGHTLY_WHEEL_INDEX
            : env.STABLE_WHEEL_INDEX;
        const wheelUrl =
          releaseType === "nightly"
            ? `${wheelIndex}/${wheel.package}/`
            : `${wheelIndex.replace(/\/simple$/, "/project")}/${wheel.package}/${env.WHEEL_VERSION}/`;
        lines.push(`• <${wheelUrl}|${wheel.package}: ${env.WHEEL_VERSION}>`);
      }
    }
    if (containerIds.length > 0) {
      lines.push("*:docker_: Containers published:*");
      for (const containerId of containerIds) {
        const container =
          releaseType === "stable"
            ? `<${env.NGC_CATALOG_BASE}/containers/${containerId}|${containerId}>`
            : containerId;
        lines.push(`• ${container}: ${env.RELEASE_LABEL}`);
      }
    }
    if (env.INCLUDE_HELM === "true") {
      const chart =
        releaseType === "stable"
          ? `<${env.NGC_CATALOG_BASE}/helm-charts/nemo-helix|nemo-helix>`
          : "nemo-helix";
      lines.push("*:helm: Helm chart published:*");
      lines.push(`• ${chart}: ${env.CHART_VERSION}`);
    }
    if (stagesNightlyWheels) {
      lines.push("", "*:python: Wheel staging dispatched:*");
      for (const wheelId of wheelIds) {
        const wheel = wheelCatalog.find(
          (candidate) => candidate.id === wheelId,
        );
        lines.push(`• ${wheel.package}: ${env.WHEEL_VERSION}`);
      }
    }
  } else {
    lines.push(
      "",
      "*Final release status:*",
      `Final artifact poll: ${env.POLL_RESULT}`,
      `GitHub release: ${env.GITHUB_RELEASE_RESULT}`,
      `Deployment signal: ${env.DEPLOYMENT_RESULT}`,
    );
  }

  lines.push("", `:link: <${env.RUN_URL}|Release run #${env.RUN_NUMBER}>`);
  return fetchImpl(webhook, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: lines.join("\n") }),
  });
}

module.exports = { sendReleaseNotification };
