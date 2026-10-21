// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { useOwnedFirstFilesets } from '@studio/components/DatasetsTable/useOwnedFirstFilesets';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { type PropsWithChildren } from 'react';

const ME = 'me@example.com';
const fileset = (name: string, created_by: string) =>
  ({ id: name, name, workspace: 'default', purpose: 'dataset', created_by }) as FilesetOutput;
const mine = ['m1', 'm2', 'm3', 'm4', 'm5'].map((n) => fileset(n, ME));
const others = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6'].map((n) => fileset(n, 'someone@example.com'));

const wrapper = ({ children }: PropsWithChildren) => <TestProviders>{children}</TestProviders>;

describe('useOwnedFirstFilesets', () => {
  beforeEach(() => {
    server.use(
      http.get(
        `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`,
        ({ request }) => {
          const params = new URL(request.url).searchParams;
          const list = params.get('filter[created_by]') === ME ? mine : others;
          const page = Number(params.get('page'));
          const size = Number(params.get('page_size'));
          return HttpResponse.json({
            data: list.slice((page - 1) * size, page * size),
            pagination: { total_results: list.length },
          });
        }
      )
    );
  });

  it('lists the user’s filesets first, then everyone else, across page boundaries', async () => {
    const pageNames = async (page: number) => {
      const { result } = renderHook(
        () =>
          useOwnedFirstFilesets({
            workspace: 'default',
            ownerId: ME,
            page,
            pageSize: 4,
            sort: undefined,
            filter: {},
          }),
        { wrapper }
      );
      await waitFor(() => expect(result.current.data).toBeDefined());
      return {
        names: result.current.data?.data.map((d) => d.name),
        total: result.current.data?.pagination?.total_results,
      };
    };

    // all owned
    expect(await pageNames(1)).toEqual({ names: ['m1', 'm2', 'm3', 'm4'], total: 11 });
    // owned runs out mid-page, others start at their first row
    expect(await pageNames(2)).toEqual({ names: ['m5', 'o1', 'o2', 'o3'], total: 11 });
    // all others, starting mid-way through the second list's first page (a straddled window)
    expect(await pageNames(3)).toEqual({ names: ['o4', 'o5', 'o6'], total: 11 });
  });
});
