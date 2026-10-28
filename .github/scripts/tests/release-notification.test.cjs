// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const { sendReleaseNotification } = require("../release-notification.cjs");

function releaseEnv(overrides = {}) {
  return {
    RELEASE_TYPE: "rc",
    RELEASE_LABEL: "rc-20261026225043",
    SOURCE_SHA: "abcdef1234567890",
    COMMIT_URL: "https://example.test/commit/abcdef1234567890",
    WHEEL_IDS: '["nemo-helix"]',
    WHEEL_CATALOG: '[{"id":"nemo-helix","package":"nemo-helix"}]',
    CONTAINER_IDS: '["nhx-api"]',
    INCLUDE_HELM: "true",
    CHART_VERSION: "0.9.0-rc-20261026225043",
    CONTAINER_REGISTRY: "ghcr.io/example/helix",
    RC_WHEEL_INDEX: "https://example.test/rc",
    STABLE_WHEEL_INDEX: "https://example.test/simple",
    PUBLISH_RC_WHEELS: "true",
    WHEEL_VERSION: "0.9.0.dev20261026225043",
    NGC_CATALOG_BASE: "https://example.test/catalog",
    POLL_RESULT: "success",
    GITHUB_RELEASE_RESULT: "success",
    DEPLOYMENT_RESULT: "success",
    RC_INSTANCE_URL: "https://rc.example.test",
    SLACK_RELEASE_WEBHOOK: "https://example.test/release-webhook",
    SLACK_ALERTS_WEBHOOK: "https://example.test/alerts-webhook",
    RUN_URL: "https://example.test/actions/runs/123",
    RUN_NUMBER: "123",
    ...overrides,
  };
}

async function notification(env) {
  let captured;
  await sendReleaseNotification({
    env,
    fetchImpl: async (url, request) => {
      captured = { url, ...JSON.parse(request.body) };
      return { ok: true };
    },
  });
  return captured;
}

test("RC blocks preserve every artifact, version and URL in the compact layout", async () => {
  const containers = [
    "nhx-api",
    "nhx-tasks",
    "nhx-tasks-openshell",
    "nhx-gym-tasks",
    "nhx-gym-host",
    "nhx-customizer-tasks",
    "nhx-automodel-training",
    "nhx-unsloth-training",
    "nhx-rl-training",
    "nhx-garak-tasks",
    "nhx-safe-synthesizer-tasks",
    "nhx-builder-tasks",
    "nhx-kaniko",
  ];
  const env = releaseEnv({ CONTAINER_IDS: JSON.stringify(containers) });
  const { url, text, blocks } = await notification(env);
  assert.equal(url, env.SLACK_RELEASE_WEBHOOK);
  assert.equal(blocks[0].text.text, "NeMo Helix · Release candidate");
  assert.equal(blocks[1].fields[0].text, `*Build*\n\`${env.RELEASE_LABEL}\``);
  assert.deepEqual(
    blocks.find((b) => b.type === "actions").elements.map((b) => b.url),
    [env.RC_INSTANCE_URL, env.RUN_URL, env.COMMIT_URL],
  );
  const wheels = blocks.find((b) => b.text?.text.startsWith(":python:"));
  assert.equal(
    wheels.text.text,
    `:python: *Python wheels* · \`${env.WHEEL_VERSION}\`\n<https://example.test/rc/nemo-helix/|nemo-helix>`,
  );
  const images = blocks.find((b) => b.type === "container");
  assert.equal(images.title, undefined);
  const heading = images.rich_text_title.elements[0].elements;
  assert.deepEqual(heading[0], { type: "emoji", name: "docker_" });
  assert.equal(heading[1].text, " Containers · 13 images");
  assert.equal(images.is_collapsible, true);
  assert.equal(images.default_collapsed, true);
  assert.equal(images.subtitle.text, `Tag: \`${env.RELEASE_LABEL}\``);
  assert.deepEqual(
    images.child_blocks[0].fields.flatMap((f) => f.text.split("\n")).sort(),
    containers.toSorted(),
  );
  assert.equal(
    images.child_blocks[1].elements[0].text,
    `Registry: ${env.CONTAINER_REGISTRY}`,
  );
  assert.equal(
    blocks.find((b) => b.text?.text.startsWith(":helm:")).text.text,
    `:helm: *Helm chart*\nnemo-helix · \`${env.CHART_VERSION}\``,
  );
  assert.equal(
    blocks.at(-1).elements[0].text,
    "1 wheel · 13 containers · 1 chart",
  );
  for (const artifact of [
    ...containers,
    env.WHEEL_VERSION,
    env.CHART_VERSION,
    env.RUN_URL,
    env.COMMIT_URL,
    env.RC_INSTANCE_URL,
  ]) {
    assert.ok(
      text.includes(artifact),
      `accessible fallback is missing ${artifact}`,
    );
  }
  assert.doesNotMatch(
    JSON.stringify(blocks),
    /TEST PREVIEW|simulated|dry run/i,
  );
});

test("stable and numbered prerelease blocks retain their published artifact links", async () => {
  for (const [version, wheelVersion] of [
    ["0.9.0", "0.9.0"],
    ["0.9.0-rc1", "0.9.0rc1"],
  ]) {
    const env = releaseEnv({
      RELEASE_TYPE: "stable",
      RELEASE_LABEL: version,
      CHART_VERSION: version,
      WHEEL_VERSION: wheelVersion,
      CONTAINER_REGISTRY: "nvcr.io/example/helix",
      RC_INSTANCE_URL: "",
    });
    const { blocks, text, url } = await notification(env);
    assert.equal(url, env.SLACK_RELEASE_WEBHOOK);
    assert.equal(blocks[0].text.text, "NeMo Helix · Release");
    assert.equal(blocks.find((b) => b.type === "actions").elements.length, 2);
    assert.doesNotMatch(text, /RC instance:/);
    const content = JSON.stringify(blocks);
    for (const link of [
      `https://example.test/project/nemo-helix/${wheelVersion}/`,
      "https://example.test/catalog/containers/nhx-api",
      "https://example.test/catalog/helm-charts/nemo-helix",
    ]) {
      assert.ok(content.includes(link));
      assert.ok(text.includes(link));
    }
    assert.ok(content.includes(env.CONTAINER_REGISTRY));
  }
});

test("staged-only wheels are described as dispatched without publication links", async () => {
  const env = releaseEnv({
    PUBLISH_RC_WHEELS: "false",
    CONTAINER_IDS: "[]",
    INCLUDE_HELM: "false",
  });
  const { blocks, text } = await notification(env);
  assert.equal(
    blocks.find((b) => b.text?.text.startsWith(":python:")).text.text,
    `:python: *Wheel staging dispatched* · \`${env.WHEEL_VERSION}\`\nnemo-helix`,
  );
  assert.equal(blocks.at(-1).elements[0].text, "1 wheel build dispatched");
  assert.doesNotMatch(
    text,
    /Artifacts published|Wheels published|example.test\/rc\/nemo-helix/,
  );
});

test("artifact subsets omit unselected sections and keep accurate counts", async () => {
  for (const [selected, overrides, count] of [
    ["python", { CONTAINER_IDS: "[]", INCLUDE_HELM: "false" }, "1 wheel"],
    ["docker_", { WHEEL_IDS: "[]", INCLUDE_HELM: "false" }, "1 container"],
    ["helm", { WHEEL_IDS: "[]", CONTAINER_IDS: "[]" }, "1 chart"],
  ]) {
    const { blocks } = await notification(releaseEnv(overrides));
    const content = JSON.stringify(blocks);
    for (const emoji of ["python", "docker_", "helm"]) {
      const present =
        emoji === "docker_"
          ? blocks.some((block) => block.type === "container")
          : content.includes(`:${emoji}:`);
      assert.equal(present, emoji === selected);
    }
    assert.equal(blocks.at(-1).elements[0].text, count);
    for (const block of blocks) {
      for (const child of block.child_blocks ?? []) {
        for (const field of child.fields ?? []) assert.notEqual(field.text, "");
      }
    }
  }
});

test("failures retain alert routing and status details without success blocks", async () => {
  for (const result of [
    "POLL_RESULT",
    "GITHUB_RELEASE_RESULT",
    "DEPLOYMENT_RESULT",
  ]) {
    for (const status of ["failure", "cancelled"]) {
      const env = releaseEnv({ [result]: status, RC_INSTANCE_URL: "" });
      const { url, text, blocks } = await notification(env);
      assert.equal(url, env.SLACK_ALERTS_WEBHOOK);
      assert.equal(blocks, undefined);
      assert.match(text, /RC release publish failed/);
      assert.ok(text.includes(`Final artifact poll: ${env.POLL_RESULT}`));
      assert.ok(text.includes(`GitHub release: ${env.GITHUB_RELEASE_RESULT}`));
      assert.ok(text.includes(`Deployment signal: ${env.DEPLOYMENT_RESULT}`));
      assert.doesNotMatch(text, /RC instance:|Artifacts published/);
    }
  }
});

test("successful rc notification requires the instance URL secret", async () => {
  await assert.rejects(
    sendReleaseNotification({
      env: releaseEnv({ RC_INSTANCE_URL: "  " }),
      fetchImpl: () => assert.fail("notification must not be sent"),
    }),
    /RC_INSTANCE_URL secret is required/,
  );
});

test("publication notifications do not include pending smoke status", async () => {
  for (const POLL_RESULT of ["success", "failure"]) {
    const payload = await notification(
      releaseEnv({ SMOKE_CHECKS_PENDING: "true", POLL_RESULT }),
    );
    assert.doesNotMatch(JSON.stringify(payload), /smoke checks pending/i);
  }
});
