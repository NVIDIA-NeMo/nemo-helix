// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { INTAKE_ENABLED } from '@e2e-tests/utils/environment';
import { disableAuthForTest } from '@e2e-tests/utils/pageUtils';
import { expect, test, type Page, type Route } from '@playwright/test';

const startedAt = '2026-01-01T00:00:00Z';
const session = {
  id: 'default-session',
  workspace: 'default',
  started_at: startedAt,
  ended_at: '2026-01-01T00:01:00Z',
  duration_ms: 60000,
  status: 'success',
  trace_count: 1,
  span_count: 65,
};
const trace = {
  ...session,
  id: 'default-trace',
  session_id: session.id,
  root_span_id: 'default-agent',
  name: 'default-agent',
  evaluation_context: { test_case_name: 'default-test-case' },
};
const span = {
  session_id: session.id,
  workspace: 'default',
  trace_id: trace.id,
  started_at: startedAt,
  ended_at: session.ended_at,
  ingested_at: session.ended_at,
  status: 'success',
  source: 'otel',
};
const spans = [
  {
    ...span,
    span_id: 'default-agent',
    kind: 'AGENT',
    name: 'default-agent',
    output: Array.from({ length: 40 }, (_, index) => `Result ${index + 1}`).join('\n'),
  },
  ...Array.from({ length: 64 }, (_, index) => ({
    ...span,
    span_id: `default-tool-${index + 1}`,
    parent_span_id: 'default-agent',
    kind: 'TOOL',
    tool_name: `default-search-policy-section-${index + 1}`,
    name: `default-search-policy-section-${index + 1}`,
    input: 'Find the relevant policy section.',
    output: 'The requested policy section was found.',
  })),
];
const workspace = {
  id: 'default-workspace',
  name: 'default',
  created_at: startedAt,
  updated_at: startedAt,
};
const pageOf = <T>(data: T[]) => ({
  data,
  pagination: {
    page: 1,
    page_size: 1000,
    current_page_size: data.length,
    total_pages: data.length > 0 ? 1 : 0,
    total_results: data.length,
  },
});
const fulfillJson = (route: Route, body: unknown): Promise<void> =>
  route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });

const setupApiMocks = async (page: Page): Promise<void> => {
  await page.route('**/health/live', (route) => fulfillJson(route, { status: 'live' }));
  await page.route('**/apis/**', (route) => {
    throw new Error(`Unexpected API request: ${route.request().url()}`);
  });
  await page.route('**/apis/plugins', (route) => fulfillJson(route, []));
  await page.route('**/apis/entities/v2/workspaces?*', (route) =>
    fulfillJson(route, pageOf([workspace]))
  );
  await page.route('**/apis/entities/v2/workspaces/default', (route) =>
    fulfillJson(route, workspace)
  );
  await page.route(`**/apis/intake/v2/workspaces/default/sessions/${session.id}`, (route) =>
    fulfillJson(route, session)
  );
  await page.route('**/apis/intake/v2/workspaces/default/traces**', (route) => {
    const url = new URL(route.request().url());
    return fulfillJson(route, url.pathname.endsWith('/traces') ? pageOf([trace]) : trace);
  });
  await page.route('**/apis/intake/v2/workspaces/default/spans**', (route) => {
    const url = new URL(route.request().url());
    const detail = spans.find(({ span_id: id }) => url.pathname.endsWith(`/${id}`));
    return fulfillJson(route, detail ?? pageOf(spans));
  });
  await page.route('**/apis/intake/v2/workspaces/default/annotations*', (route) =>
    fulfillJson(route, pageOf([]))
  );
  await page.route('**/apis/intake/v2/workspaces/default/experiments/default', (route) =>
    fulfillJson(route, { id: 'default-experiment', name: 'default', workspace: 'default' })
  );
  await page.route('**/apis/intake/v2/workspaces/default/evaluations?*', (route) =>
    fulfillJson(route, pageOf([]))
  );
};

test.describe('Test case scrolling', () => {
  test.skip(!INTAKE_ENABLED, 'Intake routes are disabled');

  test('can scroll through every tool call and open its details', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await setupApiMocks(page);
    await disableAuthForTest(page);
    await page.goto(
      `/workspaces/default/experiment/default/default/sessions/${session.id}?traceId=${trace.id}&spanId=default-agent`
    );
    await expect(page.getByText('Tool calls (64)', { exact: true })).toBeVisible();
    const lastTool = page.getByRole('button', { name: /default-search-policy-section-64/ });
    await expect(lastTool).not.toBeInViewport();

    // Scroll from the page margin so payload and trajectory scroll regions do not consume it.
    await page.mouse.move(1250, 600);
    await expect(async () => {
      await page.mouse.wheel(0, 400);
      await expect(lastTool).toBeInViewport({ timeout: 200 });
    }).toPass({ timeout: 10000 });

    await lastTool.click();
    await expect(page).toHaveURL(/spanId=default-tool-64/);
    const expandAll = page.getByRole('button', { name: 'Expand all', exact: true });
    await expect(async () => {
      await page.mouse.wheel(0, 300);
      await expect(expandAll).toBeInViewport({ timeout: 200 });
    }).toPass({ timeout: 10000 });
    await expandAll.click();
    const output = page.getByText('The requested policy section was found.', { exact: true });
    await expect(async () => {
      await page.mouse.wheel(0, 300);
      await expect(output).toBeInViewport({ timeout: 200 });
    }).toPass({ timeout: 10000 });
  });
});
