// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import {
  type DatasetsPage,
  useOwnedFirstFilesets,
} from '@studio/components/DatasetsTable/useOwnedFirstFilesets';
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

  it('continues with the first non-owned fileset after 47 owned ones at page size 10', async () => {
    // 47 filesets created by the user (m1..m47) and 60 by others (o1..o60), each list newest first.
    // The table is the concatenation [m1..m47, o1..o60] = 107 rows, shown 10 per page.
    const ownedRows = Array.from({ length: 47 }, (_, i) => fileset(`m${i + 1}`, ME));
    const otherRows = Array.from({ length: 60 }, (_, i) =>
      fileset(`o${i + 1}`, 'someone@example.com')
    );
    const requests: { owned: boolean; page: number }[] = [];
    server.use(
      http.get(
        `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`,
        ({ request }) => {
          const params = new URL(request.url).searchParams;
          const owned = params.get('filter[created_by]') === ME;
          const list = owned ? ownedRows : otherRows;
          const page = Number(params.get('page'));
          const size = Number(params.get('page_size'));
          requests.push({ owned, page });
          return HttpResponse.json({
            data: list.slice((page - 1) * size, page * size),
            pagination: { total_results: list.length },
          });
        }
      )
    );

    const loadPage = async (page: number) => {
      requests.length = 0;
      const { result } = renderHook(
        () =>
          useOwnedFirstFilesets({
            workspace: 'default',
            ownerId: ME,
            page,
            pageSize: 10,
            sort: undefined,
            filter: {},
          }),
        { wrapper }
      );
      await waitFor(() => expect(result.current.data).toBeDefined());
      return result.current.data!;
    };
    const names = (p: DatasetsPage) => p.data.map((d) => d.name);

    // Page 5 = table rows 41-50. The user's 5th owned page (server page 5) holds m41..m47, and
    // ownTotal = 47 leaves 10 - 7 = 3 slots, filled by the first three non-owned filesets.
    const page5 = await loadPage(5);
    expect(names(page5)).toEqual([
      'm41',
      'm42',
      'm43',
      'm44',
      'm45',
      'm46',
      'm47',
      'o1',
      'o2',
      'o3',
    ]);
    // Combined total is the sum of both lists, so the pager shows 107 rows.
    expect(page5.pagination?.total_results).toBe(107);

    // Page 6 = table rows 51-60. The owned list has nothing left, so the owned request returns an
    // empty page, but its total (47) still tells us where the non-owned list resumes:
    //   start = offset - ownTotal = 50 - 47 = 3  -> o4 is the first non-owned row shown here,
    // directly after o3 on page 5. There is no offset parameter, so the hook fetches whole server
    // pages of 10: start 3 lands in server page 1 (o1..o10) at skip 3, and since 3 + 10 > 10 the
    // window straddles into server page 2 (o11..o20). It keeps rows [3, 13) = o4..o13.
    const page6 = await loadPage(6);
    expect(names(page6)).toEqual(['o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11', 'o12', 'o13']);
    // Three requests: the (empty) owned page, and non-owned server pages 1 and 2.
    expect(requests).toEqual([
      { owned: true, page: 6 },
      { owned: false, page: 1 },
      { owned: false, page: 2 },
    ]);

    // Page 7 = rows 61-70: start = 60 - 47 = 13 -> skip 3 within server page 2 (o11..o20), again
    // straddling into server page 3, so the rows are o14..o23.
    const page7 = await loadPage(7);
    expect(names(page7)).toEqual([
      'o14',
      'o15',
      'o16',
      'o17',
      'o18',
      'o19',
      'o20',
      'o21',
      'o22',
      'o23',
    ]);
  });
});
