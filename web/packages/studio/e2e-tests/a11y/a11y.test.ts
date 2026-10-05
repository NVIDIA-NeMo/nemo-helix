// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { runAxeScan } from '@e2e-tests/a11y/axe';
import { disableAuthForTest } from '@e2e-tests/utils/pageUtils';
import { expect, test, type Page, type Route } from '@playwright/test';

// ---------------------------------------------------------------------------
// Fixed mock data — no live backend required.
// All API calls are intercepted by page.route() before any navigation.
// ---------------------------------------------------------------------------

const MOCK_WORKSPACE = {
  id: 'mock-workspace-id',
  name: 'default',
  description: 'Mock workspace for accessibility tests',
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
};

const fulfillJson = (route: Route, body: unknown) =>
  route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });

/**
 * Intercept NHX API calls and return fixed mock responses.
 *
 * Playwright uses LIFO for route handlers, so the workspace-specific route is
 * registered last and therefore runs before the catch-all.
 */
const setupApiMocks = async (page: Page): Promise<void> => {
  // Catch-all: empty success for any unmatched API route
  await page.route('**/apis/**', (route) => fulfillJson(route, {}));

  // WorkspaceProvider calls this on every authenticated page load
  await page.route('**/apis/entities/v2/workspaces/default', (route) =>
    fulfillJson(route, MOCK_WORKSPACE)
  );
};

test.describe('Accessibility — Studio routes (axe / WCAG 2.x A+AA)', () => {
  test.beforeEach(async ({ page }) => {
    await setupApiMocks(page);
    await disableAuthForTest(page);
  });

  test('workspace dashboard has no axe violations', async ({ page }) => {
    await page.goto('workspaces/default/dashboard');
    await page.waitForLoadState('networkidle');

    const results = await runAxeScan(page);

    expect(
      results.violations,
      `axe violations on /workspaces/default/dashboard:\n${formatViolations(results.violations)}`
    ).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface AxeViolation {
  id: string;
  description: string;
  nodes: Array<{ html: string }>;
}

const formatViolations = (violations: AxeViolation[]): string =>
  violations
    .map(
      (v) =>
        `  [${v.id}] ${v.description}\n` +
        v.nodes.map((n) => `    → ${n.html.slice(0, 120)}`).join('\n')
    )
    .join('\n');
