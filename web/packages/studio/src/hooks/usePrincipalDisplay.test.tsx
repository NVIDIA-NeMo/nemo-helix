// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ProfileResolveRequest } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { usePrincipalDisplay } from '@studio/hooks/usePrincipalDisplay';
import { server } from '@studio/mocks/node';
import { wrapper } from '@studio/tests/util/TestQueryClient';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

describe('usePrincipalDisplay', () => {
  it('batches lookups made together into one request', async () => {
    const requests: string[][] = [];
    server.use(
      http.post(`${PLATFORM_BASE_URL}/apis/auth/v2/profiles/resolve`, async ({ request }) => {
        const { principals } = (await request.json()) as ProfileResolveRequest;
        requests.push(principals);
        return HttpResponse.json({
          data: principals.map((principal) => ({
            principal,
            display_name: principal.toUpperCase(),
            display_name_is_override: false,
            ambiguous: false,
          })),
        });
      })
    );

    const { result } = renderHook(
      () => [usePrincipalDisplay('alice'), usePrincipalDisplay('bob'), usePrincipalDisplay(null)],
      { wrapper }
    );

    await waitFor(() => expect(result.current[1].data?.display_name).toBe('BOB'));
    expect(result.current[0].data?.display_name).toBe('ALICE');
    expect(result.current[2].fetchStatus).toBe('idle');
    expect(requests).toEqual([['alice', 'bob']]);
  });
});
