// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { useOwnedFirstDatasets } from '@studio/hooks/useOwnedFirstDatasets';
import { server } from '@studio/mocks/node';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { type PropsWithChildren } from 'react';

const ME = 'me@example.com';
vi.mock('@studio/providers/auth/useAuthProfile', () => ({ useAuthProfile: () => ({ id: ME }) }));

const dataset = (name: string, created_by: string) =>
  ({ id: name, name, workspace: 'default', purpose: 'dataset', created_by }) as FilesetOutput;

const wrapper = ({ children }: PropsWithChildren) => <TestProviders>{children}</TestProviders>;

describe('useOwnedFirstDatasets', () => {
  it('keeps the user’s older datasets even when they fall outside the newest page', async () => {
    // Newest first. The workspace holds 5 datasets but a picker page only fits 3, so the user's
    // oldest dataset ("mine-old") is not in the page of everyone's datasets.
    const everyone = [
      dataset('other-1', 'x'),
      dataset('mine-new', ME),
      dataset('other-2', 'x'),
      dataset('other-3', 'x'),
      dataset('mine-old', ME),
    ];
    server.use(
      http.get(
        `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`,
        ({ request }) => {
          const params = new URL(request.url).searchParams;
          const pageSize = Number(params.get('page_size'));
          const owner = params.get('filter[created_by]');
          const rows = owner ? everyone.filter((d) => d.created_by === owner) : everyone;
          return HttpResponse.json({
            data: rows.slice(0, pageSize),
            pagination: { total_results: rows.length },
          });
        }
      )
    );

    const { result } = renderHook(() => useOwnedFirstDatasets('default', 3), { wrapper });
    await waitFor(() => expect(result.current.datasets).toBeDefined());

    // Owned first (newest first), then the rest of the newest page, with no duplicates.
    expect(result.current.datasets?.map((d) => d.name)).toEqual([
      'mine-new',
      'mine-old',
      'other-1',
      'other-2',
    ]);
  });
});
