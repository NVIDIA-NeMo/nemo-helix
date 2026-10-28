// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

async function sendReleaseNotification({ env, fetchImpl }) {
  const releaseType = env.RELEASE_TYPE;
  const wheelIds = JSON.parse(env.WHEEL_IDS);
  const wheelCatalog = JSON.parse(env.WHEEL_CATALOG);
  const containerIds = JSON.parse(env.CONTAINER_IDS);
  const stagesRcWheels =
    releaseType === "rc" &&
    env.PUBLISH_RC_WHEELS !== "true" &&
    wheelIds.length > 0;
  const hasPublishedArtifacts =
    (wheelIds.length > 0 && !stagesRcWheels) ||
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
      : "*:test_tube: RC release complete*"
    : releaseType === "stable"
      ? "*:alert: Release publish failed*"
      : "*:alert: RC release publish failed*";
  const lines = [
    title,
    `Release: ${env.RELEASE_LABEL}`,
    `Commit: <${env.COMMIT_URL}|${env.SOURCE_SHA.slice(0, 7)}>`,
  ];
  const buttons = [
    {
      type: "button",
      text: { type: "plain_text", text: "View Run" },
      url: env.RUN_URL,
      action_id: "release_view_run",
    },
    {
      type: "button",
      text: { type: "plain_text", text: "Source Commit" },
      url: env.COMMIT_URL,
      action_id: "release_source_commit",
    },
  ];
  const blocks = [
    {
      type: "header",
      text: {
        type: "plain_text",
        text: `NeMo Helix · ${releaseType === "rc" ? "Release candidate" : "Release"}`,
      },
    },
    {
      type: "section",
      fields: [{ type: "mrkdwn", text: `*Build*\n\`${env.RELEASE_LABEL}\`` }],
    },
    { type: "actions", elements: buttons },
    { type: "divider" },
  ];
  const counts = [];

  if (published && releaseType === "rc") {
    const rcInstanceUrl = env.RC_INSTANCE_URL?.trim();
    if (!rcInstanceUrl) {
      throw new Error(
        "RC_INSTANCE_URL secret is required for rc notifications",
      );
    }
    lines.push(`RC instance: <${rcInstanceUrl}>`);
    buttons.unshift({
      type: "button",
      text: { type: "plain_text", text: "Open RC Instance" },
      url: rcInstanceUrl,
      action_id: "release_open_instance",
      style: "primary",
    });
  }

  if (published) {
    if (hasPublishedArtifacts) {
      lines.push("", "*Artifacts published:*");
    }
    if (wheelIds.length > 0 && !stagesRcWheels) {
      lines.push("*:python: Wheels published:*");
      const wheels = [];
      for (const wheelId of wheelIds) {
        const wheel = wheelCatalog.find(
          (candidate) => candidate.id === wheelId,
        );
        const wheelIndex =
          releaseType === "rc" ? env.RC_WHEEL_INDEX : env.STABLE_WHEEL_INDEX;
        const wheelUrl =
          releaseType === "rc"
            ? `${wheelIndex}/${wheel.package}/`
            : `${wheelIndex.replace(/\/simple$/, "/project")}/${wheel.package}/${env.WHEEL_VERSION}/`;
        lines.push(`• <${wheelUrl}|${wheel.package}: ${env.WHEEL_VERSION}>`);
        wheels.push(`<${wheelUrl}|${wheel.package}>`);
      }
      blocks.push({
        type: "section",
        text: {
          type: "mrkdwn",
          text: `:python: *Python wheels* · \`${env.WHEEL_VERSION}\`\n${wheels.join("  ·  ")}`,
        },
      });
      counts.push(
        `${wheelIds.length} ${wheelIds.length === 1 ? "wheel" : "wheels"}`,
      );
    }
    if (containerIds.length > 0) {
      lines.push("*:docker_: Containers published:*");
      const containers = [];
      for (const containerId of containerIds) {
        const container =
          releaseType === "stable"
            ? `<${env.NGC_CATALOG_BASE}/containers/${containerId}|${containerId}>`
            : containerId;
        lines.push(`• ${container}: ${env.RELEASE_LABEL}`);
        containers.push(container);
      }
      lines.push(`Registry: ${env.CONTAINER_REGISTRY}`);
      blocks.push({
        type: "container",
        rich_text_title: {
          type: "rich_text",
          elements: [
            {
              type: "rich_text_section",
              elements: [
                { type: "emoji", name: "docker_" },
                {
                  type: "text",
                  text: ` Containers · ${containerIds.length} ${containerIds.length === 1 ? "image" : "images"}`,
                  style: { bold: true },
                },
              ],
            },
          ],
        },
        subtitle: { type: "mrkdwn", text: `Tag: \`${env.RELEASE_LABEL}\`` },
        is_collapsible: true,
        default_collapsed: true,
        child_blocks: [
          {
            type: "section",
            fields: [0, 1]
              .map((column) =>
                containers.filter((_, index) => index % 2 === column),
              )
              .filter((column) => column.length > 0)
              .map((column) => ({
                type: releaseType === "stable" ? "mrkdwn" : "plain_text",
                text: column.join("\n"),
              })),
          },
          {
            type: "context",
            elements: [
              { type: "mrkdwn", text: `Registry: ${env.CONTAINER_REGISTRY}` },
            ],
          },
        ],
      });
      counts.push(
        `${containerIds.length} ${containerIds.length === 1 ? "container" : "containers"}`,
      );
    }
    if (env.INCLUDE_HELM === "true") {
      const chart =
        releaseType === "stable"
          ? `<${env.NGC_CATALOG_BASE}/helm-charts/nemo-helix|nemo-helix>`
          : "nemo-helix";
      lines.push("*:helm: Helm chart published:*");
      lines.push(`• ${chart}: ${env.CHART_VERSION}`);
      blocks.push({
        type: "section",
        text: {
          type: "mrkdwn",
          text: `:helm: *Helm chart*\n${chart} · \`${env.CHART_VERSION}\``,
        },
      });
      counts.push("1 chart");
    }
    if (stagesRcWheels) {
      lines.push("", "*:python: Wheel staging dispatched:*");
      const wheels = [];
      for (const wheelId of wheelIds) {
        const wheel = wheelCatalog.find(
          (candidate) => candidate.id === wheelId,
        );
        lines.push(`• ${wheel.package}: ${env.WHEEL_VERSION}`);
        wheels.push(wheel.package);
      }
      blocks.push({
        type: "section",
        text: {
          type: "mrkdwn",
          text: `:python: *Wheel staging dispatched* · \`${env.WHEEL_VERSION}\`\n${wheels.join("  ·  ")}`,
        },
      });
      counts.push(
        `${wheelIds.length} wheel ${wheelIds.length === 1 ? "build" : "builds"} dispatched`,
      );
    }
    if (counts.length > 0) {
      blocks.push({
        type: "context",
        elements: [{ type: "mrkdwn", text: counts.join(" · ") }],
      });
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
    body: JSON.stringify({
      text: lines.join("\n"),
      ...(published ? { blocks } : {}),
    }),
  });
}

module.exports = { sendReleaseNotification };
