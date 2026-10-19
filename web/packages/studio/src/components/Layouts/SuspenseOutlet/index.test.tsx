// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SuspenseOutlet } from '@studio/components/Layouts/SuspenseOutlet';
import { act, render, screen } from '@testing-library/react';
import { use, useEffect } from 'react';
import { createMemoryRouter, useParams } from 'react-router';
import { RouterProvider } from 'react-router/dom';

const pending = new Promise<never>(() => {});

const Suspends = () => {
  use(pending);
  return null;
};

const Item = () => {
  const { id } = useParams();
  if (id === 'pending') {
    return <Suspends />;
  }
  return <div>Item {id}</div>;
};

const onListMount = vi.fn();

const List = () => {
  const { id } = useParams();
  useEffect(() => onListMount(), []);
  return <div>List {id ?? 'all'}</div>;
};

const mountRouter = (initialPath: string) => {
  const router = createMemoryRouter(
    [
      {
        element: <SuspenseOutlet fallback={<div>fallback</div>} />,
        children: [
          { path: '/a', element: <div>Page A</div> },
          { path: '/b', element: <Suspends /> },
          { path: '/items/:id', element: <Item /> },
          {
            element: <List />,
            children: [
              { path: '/list', element: null },
              { path: '/list/:id', element: null },
            ],
          },
        ],
      },
    ],
    { initialEntries: [initialPath] }
  );
  render(<RouterProvider router={router} />);
  return router;
};

describe('SuspenseOutlet', () => {
  it('shows the fallback immediately when navigating to a different child route that suspends', async () => {
    const router = mountRouter('/a');
    expect(await screen.findByText('Page A')).toBeInTheDocument();

    await act(() => router.navigate('/b'));

    expect(screen.getByText('fallback')).toBeInTheDocument();
    expect(screen.queryByText('Page A')).not.toBeInTheDocument();
  });

  it('keeps the current content visible when the same child route suspends on a param change', async () => {
    const router = mountRouter('/items/1');
    expect(await screen.findByText('Item 1')).toBeInTheDocument();

    await act(() => router.navigate('/items/pending'));

    expect(screen.getByText('Item 1')).toBeInTheDocument();
    expect(screen.queryByText('fallback')).not.toBeInTheDocument();
  });

  it('keeps a pathless route element mounted when moving between its child paths', async () => {
    onListMount.mockClear();
    const router = mountRouter('/list');
    expect(await screen.findByText('List all')).toBeInTheDocument();

    await act(() => router.navigate('/list/1'));

    expect(screen.getByText('List 1')).toBeInTheDocument();
    expect(onListMount).toHaveBeenCalledTimes(1);
  });
});
