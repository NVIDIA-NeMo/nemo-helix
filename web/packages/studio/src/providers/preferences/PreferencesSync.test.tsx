// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { PreferencesSync } from '@studio/providers/preferences/PreferencesSync';
import { wrapper } from '@studio/tests/util/TestQueryClient';
import { TOUR_SEEN_KEY, UI_THEME } from '@studio/util/localStorage';
import { render, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

vi.mock('@studio/constants/featureFlags', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@studio/constants/featureFlags')>();
  return { featureFlags: { ...actual.featureFlags, serverPreferencesEnabled: true } };
});
vi.mock('@studio/providers/auth/useMe', () => ({
  useMe: () => ({ data: { account_id: 'account-1' }, isPending: false }),
}));

describe('PreferencesSync legacy migration', () => {
  it('only fills the slot the read saw empty, and leaves a concurrent write alone', async () => {
    window.localStorage.setItem(UI_THEME, JSON.stringify('light'));
    window.localStorage.setItem(TOUR_SEEN_KEY, JSON.stringify(true));
    const ifMatch: Record<string, string | null> = {};
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/auth/v2/me/preferences/:namespace`, ({ params }) =>
        HttpResponse.json({
          namespace: params.namespace,
          value: {},
          defaults: {},
          effective: {},
          db_version: params.namespace === 'studio.ui' ? 3 : 0,
        })
      ),
      http.patch(
        `${PLATFORM_BASE_URL}/apis/auth/v2/me/preferences/:namespace`,
        ({ params, request }) => {
          ifMatch[params.namespace as string] = request.headers.get('If-Match');
          // The theme changed since the read; the tour flag did not.
          return params.namespace === 'studio.ui'
            ? HttpResponse.json({ detail: 'conflict' }, { status: 412 })
            : HttpResponse.json({});
        }
      )
    );

    render(<PreferencesSync />, { wrapper });

    await waitFor(() => expect(ifMatch).toEqual({ 'studio.ui': '3', 'studio.onboarding': '0' }));
  });
});
