// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LONG_OPERATION_TIMEOUT } from '@e2e-tests/utils/constants';
import { Page, expect } from '@playwright/test';
import { TOUR_SEEN_KEY } from '@studio/util/localStorage';

export const disableAuthForTest = async (page: Page) => {
  await page.goto('./');
  await page.evaluate((tourSeenKey) => {
    localStorage.setItem('e2e_test', 'true');
    // Prevent the welcome tour from auto-starting and blocking clicks
    localStorage.setItem(tourSeenKey, 'true');
  }, TOUR_SEEN_KEY);
};

export const textExists = async (page: Page, name: string): Promise<boolean> => {
  if (await page.getByText(name).count()) {
    return true;
  }
  return false;
};

interface WaitForTaskCompletionOptions {
  page: Page;
  evaluate: () => Promise<boolean>;
  pollInterval?: number;
  timeout?: number;
}

export const waitForTaskCompletion = async ({
  evaluate,
  page,
  pollInterval = 15 * 1000, // 15 seconds
  timeout = 10 * 60 * 1000, // 10 minutes
}: WaitForTaskCompletionOptions): Promise<void> => {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const isComplete = await evaluate();

    if (isComplete) {
      return;
    }
    await page.waitForTimeout(pollInterval); // Wait before polling again
  }

  throw new Error('Task did not complete within timeout');
};

/**
 * Enhanced waiting that combines network idle + spinner disappearing.
 * Use this for robust waiting on long-running operations.
 */
export const waitForLongOperation = async (
  page: Page,
  timeout = LONG_OPERATION_TIMEOUT
): Promise<void> => {
  // Wait for network to be idle
  await page.waitForLoadState('networkidle', { timeout });

  // Wait for common spinner selectors to disappear
  const spinnerSelectors = [
    '[data-testid="spinner"]',
    '[data-testid="nv-spinner-spinner"]',
    '.loading',
    '.spinner',
  ];

  for (const selector of spinnerSelectors) {
    const locator = page.locator(selector);
    // Only wait if the spinner actually exists
    const count = await locator.count();
    if (count > 0) {
      try {
        await expect(locator).not.toBeVisible({ timeout });
      } catch {
        // Ignore if spinner doesn't disappear in time
      }
    }
  }
};

/**
 * If a toast/notification containing the provided text exists, assert it is visible.
 */
export const expectToastIsVisible = async (
  page: Page,
  text: string | RegExp,
  timeout: number = LONG_OPERATION_TIMEOUT
): Promise<void> =>
  await expect(page.getByTestId('nv-toast-root').filter({ hasText: text })).toBeVisible({
    timeout,
  });
