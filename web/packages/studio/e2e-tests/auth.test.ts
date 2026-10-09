// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { STUDIO_URL_BASE_PATH } from '@e2e-tests/utils/environment';
import { expect, type Page, test } from '@playwright/test';
import { TOUR_SEEN_KEY } from '@studio/util/localStorage';

const AUTH_EXPLICIT_LOGOUT_STORAGE_KEY = 'nemo.studio.auth.explicitLogout';

const serverSessionDiscovery = {
  auth_enabled: true,
  oidc: {
    clients: [{ name: 'confidential', client_authentication: 'client_secret_basic' }],
  },
};

const studioRootPath = STUDIO_URL_BASE_PATH.endsWith('/')
  ? STUDIO_URL_BASE_PATH
  : `${STUDIO_URL_BASE_PATH}/`;

const suppressWelcomeTour = async (page: Page): Promise<void> => {
  await page.addInitScript((tourSeenKey) => {
    window.localStorage.setItem(tourSeenKey, 'true');
  }, TOUR_SEEN_KEY);
};

test.describe('Authentication', () => {
  test('shows a sign-in action at root after explicit server-session logout', async ({ page }) => {
    let loginUrl: string | undefined;

    await suppressWelcomeTour(page);
    await page.addInitScript((explicitLogoutKey) => {
      window.sessionStorage.setItem(explicitLogoutKey, 'true');
    }, AUTH_EXPLICIT_LOGOUT_STORAGE_KEY);
    await page.route('**/apis/auth/discovery', async (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(serverSessionDiscovery),
      })
    );
    await page.route('**/apis/auth/v2/session', async (route) => route.fulfill({ status: 401 }));
    await page.route('**/apis/auth/v2/login?**', async (route) => {
      loginUrl = route.request().url();
      await route.fulfill({ status: 200, contentType: 'text/plain', body: 'login' });
    });

    await page.goto('./');

    await expect(page.getByRole('heading', { name: 'Signed out' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Sign in' })).toBeVisible();

    await page.getByRole('button', { name: 'Sign in' }).click();
    await expect.poll(() => loginUrl).toBeTruthy();

    const login = new URL(loginUrl ?? '');
    expect(login.pathname).toBe('/apis/auth/v2/login');
    expect(login.searchParams.get('client')).toBe('confidential');
    expect(login.searchParams.get('return_to')).toBe(studioRootPath);
  });

  test('loads Studio when auth discovery is unavailable', async ({ page }) => {
    await suppressWelcomeTour(page);
    await page.route('**/apis/auth/discovery', async (route) => route.fulfill({ status: 404 }));

    await page.goto('./');

    await expect(page.getByText('NeMo Studio')).toBeVisible();
    await expect(page.getByText('Signed out')).toHaveCount(0);
  });
});
