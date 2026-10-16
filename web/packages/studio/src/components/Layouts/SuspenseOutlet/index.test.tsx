// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SuspenseOutlet } from '@studio/components/Layouts/SuspenseOutlet';
import { act, render, screen } from '@testing-library/react';
import { use } from 'react';
import { createMemoryRouter, useParams } from 'react-router';
import { RouterProvider } from 'react-router/dom';

const pending = new Promise<never>(() => {});

const Suspends = () => {
  use(pending);
  return null;
};

const Item = () => {
  const { id } = useParams();
  if (id === 'pending') return <Suspends />;
  return <div>Item {id}</div>;
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
});
