// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PreferenceDocument } from '@nemo/sdk/generated/platform/schema';
import {
  enqueuePreferencePatch,
  PREFERENCE_WRITE_DEBOUNCE_MS,
} from '@studio/providers/preferences/writeQueue';
import { QueryClient } from '@tanstack/react-query';

const { mockPatch } = vi.hoisted(() => ({ mockPatch: vi.fn() }));

vi.mock('@nemo/sdk/generated/platform/profiles', () => ({
  authPatchMyPreference: mockPatch,
  getAuthGetMyPreferenceQueryKey: (namespace: string) => [namespace],
}));

const document = (theme: string): PreferenceDocument => ({
  namespace: 'studio.ui',
  value: { theme },
  defaults: {},
  effective: { theme },
  db_version: 1,
});

describe('enqueuePreferencePatch', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('ignores a response that arrives after a newer request', async () => {
    const resolvers: ((document: PreferenceDocument) => void)[] = [];
    mockPatch.mockImplementation(() => new Promise((resolve) => resolvers.push(resolve)));
    const queryClient = new QueryClient({ defaultOptions: { queries: { gcTime: Infinity } } });
    const write = { accountId: 'a', namespace: 'studio.ui', queryClient };

    enqueuePreferencePatch({ ...write, patch: { theme: 'light' } });
    await vi.advanceTimersByTimeAsync(PREFERENCE_WRITE_DEBOUNCE_MS);
    enqueuePreferencePatch({ ...write, patch: { theme: 'system' } });
    await vi.advanceTimersByTimeAsync(PREFERENCE_WRITE_DEBOUNCE_MS);
    expect(resolvers).toHaveLength(2);

    resolvers[1](document('system'));
    await vi.runAllTimersAsync();
    resolvers[0](document('light'));
    await vi.runAllTimersAsync();

    expect(queryClient.getQueryData(['studio.ui'])).toEqual(document('system'));
  });
});
