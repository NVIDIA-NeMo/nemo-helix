// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PreferenceDocument } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { studioPreferences } from '@studio/providers/preferences/definitions';
import { MIRROR_PREFIX } from '@studio/providers/preferences/mirror';
import { usePreference } from '@studio/providers/preferences/usePreference';
import { wrapper } from '@studio/tests/util/TestQueryClient';
import { act, renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

vi.mock('@studio/constants/featureFlags', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@studio/constants/featureFlags')>();
  return { ...actual, featureFlags: { ...actual.featureFlags, serverPreferencesEnabled: true } };
});

const SIGNED_IN = { data: { account_id: 'account-1' }, isPending: false };
const { mockMe } = vi.hoisted(() => ({ mockMe: vi.fn() }));
vi.mock('@studio/providers/auth/useMe', () => ({ useMe: mockMe }));

const document = (namespace: string, value: Record<string, unknown>, defaults = {}) =>
  ({
    namespace,
    value,
    defaults,
    effective: { ...defaults, ...value },
    db_version: 1,
  }) satisfies PreferenceDocument;

describe('usePreference with server preferences', () => {
  beforeEach(() => {
    window.localStorage.clear();
    mockMe.mockReturnValue(SIGNED_IN);
  });

  it('reads the effective value and falls back to the default when it is invalid', async () => {
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/auth/v2/me/preferences/studio.ui`, () =>
        HttpResponse.json(
          document('studio.ui', { recentWorkspaces: 'not-a-list' }, { theme: 'light' })
        )
      )
    );
    const { result } = renderHook(
      () => [
        usePreference(studioPreferences.theme),
        usePreference(studioPreferences.recentWorkspaces),
      ],
      { wrapper }
    );

    await waitFor(() => expect(result.current[0][3]).toBe(true));
    expect(result.current[0][0]).toBe('light');
    expect(result.current[1][0]).toEqual([]);
    expect(window.localStorage.getItem(`${MIRROR_PREFIX}account-1:studio.ui:`)).toContain('light');
  });

  it('applies writes optimistically and sends one debounced merge patch', async () => {
    const patches: unknown[] = [];
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/auth/v2/me/preferences/studio.ui`, () =>
        HttpResponse.json(document('studio.ui', {}))
      ),
      http.patch(
        `${PLATFORM_BASE_URL}/apis/auth/v2/me/preferences/studio.ui`,
        async ({ request }) => {
          const patch = (await request.json()) as Record<string, unknown>;
          patches.push(patch);
          return HttpResponse.json(document('studio.ui', patch));
        }
      )
    );
    const { result } = renderHook(
      () => ({
        theme: usePreference(studioPreferences.theme),
        recent: usePreference(studioPreferences.recentWorkspaces),
      }),
      { wrapper }
    );
    await waitFor(() => expect(result.current.theme[3]).toBe(true));

    act(() => {
      result.current.theme[1]('light');
      result.current.recent[1](['a', 'b', 'c', 'd', 'e', 'f']);
    });

    await waitFor(() => expect(result.current.theme[0]).toBe('light'));
    await waitFor(() => expect(patches).toHaveLength(1));
    expect(patches[0]).toEqual({ theme: 'light', recentWorkspaces: ['a', 'b', 'c', 'd', 'e'] });
  });

  it('evicts the oldest map entries beyond the cap', async () => {
    const entries = Object.fromEntries(
      Array.from({ length: 50 }, (_, i) => [
        `group-${i}`,
        { v: { visible: true, flag: false }, at: i },
      ])
    );
    const patches: unknown[] = [];
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/auth/v2/me/preferences/studio.experiments`, () =>
        HttpResponse.json(document('studio.experiments', { trendVisibility: entries }))
      ),
      http.patch(
        `${PLATFORM_BASE_URL}/apis/auth/v2/me/preferences/studio.experiments`,
        async ({ request }) => {
          patches.push(await request.json());
          return HttpResponse.json(document('studio.experiments', {}));
        }
      )
    );
    const { result } = renderHook(
      () => usePreference(studioPreferences.experimentTrendVisibility, { entry: 'group-new' }),
      { wrapper }
    );
    await waitFor(() => expect(result.current[3]).toBe(true));

    act(() => result.current[1]({ visible: false, flag: true }));

    await waitFor(() => expect(result.current[0]).toEqual({ visible: false, flag: true }));
    await waitFor(() => expect(patches).toHaveLength(1));
    const written = (patches[0] as { trendVisibility: Record<string, unknown> }).trendVisibility;
    expect(written['group-0']).toBeNull();
    expect(written['group-1']).toBeUndefined();
    expect(written['group-new']).toMatchObject({ v: { visible: false, flag: true } });
  });

  it('is ready with the default once the read fails', async () => {
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/auth/v2/me/preferences/studio.onboarding`, () =>
        HttpResponse.json({ detail: 'unavailable' }, { status: 503 })
      )
    );
    const { result } = renderHook(() => usePreference(studioPreferences.tourSeen), { wrapper });

    await waitFor(() => expect(result.current[3]).toBe(true));
    expect(result.current[0]).toBe(false);
  });

  it('keeps writes made before /me resolves and sends them once the account is known', async () => {
    const patches: unknown[] = [];
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/auth/v2/me/preferences/studio.ui`, () =>
        HttpResponse.json(document('studio.ui', { theme: 'dark' }))
      ),
      http.patch(
        `${PLATFORM_BASE_URL}/apis/auth/v2/me/preferences/studio.ui`,
        async ({ request }) => {
          const patch = (await request.json()) as Record<string, unknown>;
          patches.push(patch);
          return HttpResponse.json(document('studio.ui', patch));
        }
      )
    );
    mockMe.mockReturnValue({ data: undefined, isPending: true });
    const { result, rerender } = renderHook(() => usePreference(studioPreferences.theme), {
      wrapper,
    });

    act(() => result.current[1]('light'));
    await waitFor(() => expect(result.current[0]).toBe('light'));

    mockMe.mockReturnValue(SIGNED_IN);
    rerender();

    await waitFor(() => expect(patches).toEqual([{ theme: 'light' }]));
    expect(result.current[0]).toBe('light');
  });
});
