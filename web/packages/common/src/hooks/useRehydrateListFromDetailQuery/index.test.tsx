// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRehydrateListFromDetailQuery } from '@nemo/common/src/hooks/useRehydrateListFromDetailQuery/index';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { type ReactNode } from 'react';

type Row = { id: string; label: string; version?: number };

function createWrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

describe('useRehydrateListFromDetailQuery', () => {
  it('merges detail into cached list rows with the same id', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const listKey = ['test-list', 'ws-a'] as const;
    const initial: { data: Row[] } = {
      data: [
        { id: '1', label: 'one', version: 1 },
        { id: '2', label: 'two' },
      ],
    };
    client.setQueryData(listKey, initial);

    renderHook(
      () =>
        useRehydrateListFromDetailQuery<Row, Row>({
          detail: { id: '1', label: 'updated', version: 2 },
          listQueryKey: listKey,
          detailToListItem: (d) => d,
          getRowId: (r) => r.id,
        }),
      { wrapper: createWrapper(client) }
    );

    await waitFor(() => {
      expect(client.getQueryData(listKey)).toEqual({
        data: [
          { id: '1', label: 'updated', version: 2 },
          { id: '2', label: 'two' },
        ],
      });
    });
  });

  it('does not append when addIfMissing is false and row is absent', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const listKey = ['test-list', 'ws-b'] as const;
    const initial: { data: Row[] } = { data: [{ id: '1', label: 'one' }] };
    client.setQueryData(listKey, initial);

    renderHook(
      () =>
        useRehydrateListFromDetailQuery<Row, Row>({
          detail: { id: '99', label: 'new' },
          listQueryKey: listKey,
          detailToListItem: (d) => d,
          getRowId: (r) => r.id,
          addIfMissing: false,
        }),
      { wrapper: createWrapper(client) }
    );

    await waitFor(() => {
      expect(client.getQueryData(listKey)).toEqual(initial);
    });
  });

  it('appends when addIfMissing is true and row is absent', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const listKey = ['test-list', 'ws-c'] as const;
    const initial: { data: Row[] } = { data: [{ id: '1', label: 'one' }] };
    client.setQueryData(listKey, initial);

    renderHook(
      () =>
        useRehydrateListFromDetailQuery<Row, Row>({
          detail: { id: '99', label: 'new' },
          listQueryKey: listKey,
          detailToListItem: (d) => d,
          getRowId: (r) => r.id,
          addIfMissing: true,
        }),
      { wrapper: createWrapper(client) }
    );

    await waitFor(() => {
      expect(client.getQueryData(listKey)).toEqual({
        data: [
          { id: '1', label: 'one' },
          { id: '99', label: 'new' },
        ],
      });
    });
  });

  it('updates every query whose key starts with listQueryKey', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const prefix = ['/apis/items'] as const;
    client.setQueryData([...prefix, { page: 1 }] as const, {
      data: [{ id: 'a', label: 'p1' }],
    });
    client.setQueryData([...prefix, { page: 2 }] as const, {
      data: [{ id: 'a', label: 'p2-old' }],
    });

    renderHook(
      () =>
        useRehydrateListFromDetailQuery<Row, Row>({
          detail: { id: 'a', label: 'fresh' },
          listQueryKey: prefix,
          detailToListItem: (d) => d,
          getRowId: (r) => r.id,
        }),
      { wrapper: createWrapper(client) }
    );

    await waitFor(() => {
      expect(client.getQueryData([...prefix, { page: 1 }] as const)).toEqual({
        data: [{ id: 'a', label: 'fresh' }],
      });
      expect(client.getQueryData([...prefix, { page: 2 }] as const)).toEqual({
        data: [{ id: 'a', label: 'fresh' }],
      });
    });
  });
});
