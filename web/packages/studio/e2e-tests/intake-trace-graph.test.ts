// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { INTAKE_ENABLED, TRACE_GRAPH_ENABLED } from '@e2e-tests/utils/environment';
import { disableAuthForTest } from '@e2e-tests/utils/pageUtils';
import { expect, test, type Page, type Route } from '@playwright/test';

const startedAtMs = Date.now() - 5000;
const at = (offsetMs: number): string => new Date(startedAtMs + offsetMs).toISOString();

const session = {
  id: 'session-graph-test',
  workspace: 'default',
  started_at: at(0),
  ended_at: at(4000),
  duration_ms: 4000,
  status: 'success',
  trace_count: 1,
  span_count: 4,
};

const trace = {
  id: 'trace-graph-test',
  root_span_id: 'span-root',
  session_id: session.id,
  workspace: 'default',
  name: 'Answer a policy question',
  started_at: session.started_at,
  ended_at: session.ended_at,
  duration_ms: session.duration_ms,
  status: 'success',
  span_count: 4,
};

const workspace = {
  id: 'workspace-default',
  name: 'default',
  created_at: session.started_at,
  updated_at: session.started_at,
};

const spans = [
  {
    span_id: 'span-root',
    session_id: session.id,
    workspace: 'default',
    kind: 'AGENT',
    name: 'policy-agent',
    source: 'otel',
    trace_id: trace.id,
    started_at: at(0),
    ended_at: at(4000),
    status: 'success',
    ingested_at: at(5000),
  },
  ...['span-search-1', 'span-search-2'].map((spanId, index) => ({
    span_id: spanId,
    parent_span_id: 'span-root',
    session_id: session.id,
    workspace: 'default',
    kind: 'TOOL',
    name: 'search-policy',
    tool_name: 'search-policy',
    source: 'otel',
    trace_id: trace.id,
    started_at: at((index + 1) * 1000),
    ended_at: at((index + 2) * 1000),
    status: 'success',
    ingested_at: at(5000),
  })),
  {
    span_id: 'span-answer',
    parent_span_id: 'span-search-2',
    session_id: session.id,
    workspace: 'default',
    kind: 'LLM',
    name: 'write-answer',
    source: 'otel',
    trace_id: trace.id,
    started_at: at(3000),
    ended_at: at(4000),
    status: 'success',
    total_tokens: 320,
    ingested_at: at(5000),
  },
];

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
    const span = spans.find(({ span_id: spanId }) => url.pathname.endsWith(`/${spanId}`));
    return fulfillJson(route, span ?? pageOf(spans));
  });
  await page.route('**/apis/intake/v2/workspaces/default/annotations*', (route) =>
    fulfillJson(route, pageOf([]))
  );
};

const viewportZoom = async (page: Page): Promise<string | undefined> => {
  const style = await page.locator('.react-flow__viewport').getAttribute('style');
  return style?.match(/scale\(([^)]+)\)/)?.[1];
};

test.describe('Intake trace graph', () => {
  test.skip(!INTAKE_ENABLED, 'Intake routes are disabled');
  test.skip(!TRACE_GRAPH_ENABLED, 'Trace graph view is disabled');

  test.beforeEach(async ({ page }) => {
    await setupApiMocks(page);
    await disableAuthForTest(page);
  });

  test('opens a trace and keeps context while exploring its graph @record', async ({ page }) => {
    await page.goto('/workspaces/default/intake/traces');
    await page.getByText(trace.name, { exact: true }).click();
    await expect(page).toHaveURL(
      `/workspaces/default/intake/sessions/${session.id}?traceId=${trace.id}`
    );

    await page.getByText('Graph', { exact: true }).click();
    await expect(page.getByText('3 groups from 4 spans')).toBeVisible();
    await page.getByRole('button', { name: /Search Policy/ }).click();
    await expect(page.getByRole('combobox', { name: 'Selected call' })).toBeVisible();

    const separator = page.getByRole('separator', { name: 'Resize panels' });
    const initialWidth = Number(await separator.getAttribute('aria-valuenow'));
    await separator.press('ArrowRight');
    await expect(separator).toHaveAttribute('aria-valuenow', String(initialWidth + 24));

    await page.locator('.react-flow__controls-zoomin').click();
    const zoom = await viewportZoom(page);
    await page.getByText('All spans', { exact: true }).click();

    await expect(page.locator('.react-flow__node[data-id="span-search-1"] button')).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    await expect.poll(() => viewportZoom(page)).toBe(zoom);

    const longestPath = page.getByRole('button', {
      name: 'Longest path',
      exact: true,
    });
    await longestPath.click();
    await expect(longestPath).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('button[aria-label$=", longest path"]')).toHaveCount(3);

    await page.getByRole('button', { name: 'Most tokens' }).click();
    await expect(page).toHaveURL(/spanId=span-answer/);
    await expect(page.locator('.react-flow__node[data-id="span-answer"] button')).toHaveAttribute(
      'aria-pressed',
      'true'
    );
  });
});
