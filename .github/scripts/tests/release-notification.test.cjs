// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const test = require("node:test");
const { sendReleaseNotification } = require("../release-notification.cjs");

function releaseEnv(overrides = {}) {
  return {
    RELEASE_TYPE: "nightly",
    RELEASE_LABEL: "nightly-20260923",
    SOURCE_SHA: "abcdef1234567890",
    COMMIT_URL: "https://example.test/commit/abcdef1234567890",
    WHEEL_IDS: '["nemo-helix"]',
    WHEEL_CATALOG: '[{"id":"nemo-helix","package":"nemo-helix"}]',
    CONTAINER_IDS: '["nhx-api"]',
    INCLUDE_HELM: "true",
    CHART_VERSION: "0.6.0-nightly-20260923",
    NIGHTLY_WHEEL_INDEX: "https://example.test/nightly",
    STABLE_WHEEL_INDEX: "https://example.test/simple",
    PUBLISH_NIGHTLY_WHEELS: "true",
    WHEEL_VERSION: "0.6.0.dev20260923",
    NGC_CATALOG_BASE: "https://example.test/catalog",
    POLL_RESULT: "success",
    GITHUB_RELEASE_RESULT: "success",
    DEPLOYMENT_RESULT: "success",
    NIGHTLY_INSTANCE_URL: "https://nightly.example.test",
    SLACK_RELEASE_WEBHOOK: "https://example.test/release-webhook",
    SLACK_ALERTS_WEBHOOK: "https://example.test/alerts-webhook",
    RUN_URL: "https://example.test/actions/runs/123",
    RUN_NUMBER: "123",
    ...overrides,
  };
}

async function notificationText(env) {
  let text;
  await sendReleaseNotification({
    env,
    fetchImpl: async (_url, request) => {
      text = JSON.parse(request.body).text;
      return { ok: true };
    },
  });
  return text;
}

test("successful nightly notification links the instance and uses visible bullets", async () => {
  const text = await notificationText(releaseEnv());
  assert.match(
    text,
    /Commit: <[^\n]+>\nNightly instance: <https:\/\/nightly\.example\.test>/,
  );
  assert.match(text, /^• <[^\n]+\|nemo-helix: 0\.6\.0\.dev20260923>$/m);
  assert.match(text, /^• nhx-api: nightly-20260923$/m);
  assert.match(text, /^• nemo-helix: 0\.6\.0-nightly-20260923$/m);
  assert.doesNotMatch(text, /^- /m);
});

test("stable and failed notifications omit the nightly instance", async () => {
  const stable = await notificationText(releaseEnv({ RELEASE_TYPE: "stable" }));
  const failed = await notificationText(releaseEnv({ POLL_RESULT: "failure" }));
  assert.doesNotMatch(stable, /Nightly instance:/);
  assert.match(stable, /^• /m);
  assert.doesNotMatch(failed, /Nightly instance:/);
});

test("staged nightly wheels use visible bullets", async () => {
  const text = await notificationText(
    releaseEnv({ PUBLISH_NIGHTLY_WHEELS: "false" }),
  );
  assert.match(
    text,
    /Wheel staging dispatched:\*\n• nemo-helix: 0\.6\.0\.dev20260923/,
  );
});

test("successful nightly notification requires the instance URL secret", async () => {
  await assert.rejects(
    sendReleaseNotification({
      env: releaseEnv({ NIGHTLY_INSTANCE_URL: "  " }),
      fetchImpl: () => assert.fail("notification must not be sent"),
    }),
    /NIGHTLY_INSTANCE_URL secret is required/,
  );
});
