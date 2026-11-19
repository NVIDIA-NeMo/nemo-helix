// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { STUDIO_URL } from '@e2e-tests/utils/environment';
import { defineConfig, devices, type ViewportSize } from '@playwright/test';

const DESKTOP_VIEWPORT_SIZE: ViewportSize = { width: 1920, height: 1080 };

/**
 * See https://playwright.dev/docs/test-configuration.
 */
// eslint-disable-next-line import/no-default-export
export default defineConfig({
  testDir: './e2e-tests',
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  timeout: 1 * 60 * 1000, // 1 Minute
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: process.env.CI
    ? [['html'], ['junit', { outputFile: 'test-results/results.xml' }], ['list']]
    : // `list` prints per-test pass/skip results inline; `html` with open:'never' still
      // generates the report (open it with `npx playwright show-report`) but never spawns
      // a blocking report server, so a plain `pnpm test:e2e` run can't hang the terminal.
      [['list'], ['html', { open: 'never' }]],
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: STUDIO_URL,

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: process.env.RECORD ? 'on' : 'on-first-retry',

    /* Record video on failure for debugging. See https://playwright.dev/docs/videos */
    video: {
      mode: process.env.RECORD ? 'on' : 'retain-on-failure',
      size: DESKTOP_VIEWPORT_SIZE,
    },
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: DESKTOP_VIEWPORT_SIZE,
      },
    },
    // {
    //   name: 'firefox',
    //   use: {
    //     ...devices['Desktop Firefox'],
    //     viewport: DESKTOP_VIEWPORT_SIZE,
    //   },
    // },

    // {
    //   name: 'webkit',
    //   use: {
    //     ...devices['Desktop Safari'],
    //     viewport: DESKTOP_VIEWPORT_SIZE,
    //   },
    // },

    /* Test against mobile viewports. */
    // {
    //   name: 'Mobile Chrome',
    //   use: { ...devices['Pixel 5'] },
    // },
    // {
    //   name: 'Mobile Safari',
    //   use: { ...devices['iPhone 12'] },
    // },

    /* Test against branded browsers. */
    // {
    //   name: 'Microsoft Edge',
    //   use: {
    //     ...devices['Desktop Edge'],
    //     channel: 'msedge',
    //     viewport: DESKTOP_VIEWPORT_SIZE,
    //   },
    // },

    // {
    //   name: 'Google Chrome',
    //   use: {
    //     ...devices['Desktop Chrome'],
    //     channel: 'chrome',
    //     viewport: DESKTOP_VIEWPORT_SIZE,
    //   },
    // },
  ],

  /* Run your local dev server before starting the tests */
  // webServer: {
  //   command: 'npm run start',
  //   url: 'http://127.0.0.1:3000',
  //   reuseExistingServer: !process.env.CI,
  // },
});
