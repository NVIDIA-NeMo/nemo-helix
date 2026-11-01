// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel, type ErrorPanelProps } from '@nemo/common/src/components/ErrorPanel';
import { suppressConsoleError } from '@nemo/testing/utils/suppress-console';
import { mockUseNavigate } from '@studio/tests/util/mockUseParams';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AxiosError, AxiosHeaders } from 'axios';
import { createMemoryRouter, RouterProvider, useNavigate } from 'react-router';

// React error boundaries always call console.error when catching errors.
beforeEach(() => {
  suppressConsoleError(
    'The above error occurred in',
    'React Router caught the following error',
    'undefined'
  );
});

// mockUseNavigate replaces the implementation for the rest of the file unless reset.
afterEach(() => vi.mocked(useNavigate).mockReset());

const axiosError = (status: number, detail?: string) =>
  new AxiosError('Request failed', undefined, undefined, undefined, {
    status,
    statusText: '',
    data: detail ? { detail } : undefined,
    headers: {},
    config: { headers: new AxiosHeaders() },
  });

// Renders a route that throws `error`, with RouteErrorPanel as its errorElement.
const renderWithRouterError = (
  props: ErrorPanelProps,
  error: Error | { status: number; statusText: string },
  queryClient = new QueryClient()
) => {
  let shouldThrow = true;
  const ThrowingComponent = () => {
    if (!shouldThrow) return <div>Recovered</div>;
    if (error instanceof Error) throw error;
    throw new Response(null, { status: error.status, statusText: error.statusText });
  };

  const router = createMemoryRouter(
    [{ path: '/', element: <ThrowingComponent />, errorElement: <RouteErrorPanel {...props} /> }],
    { initialEntries: ['/'] }
  );

  render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
  return {
    stopThrowing: () => {
      shouldThrow = false;
    },
  };
};

describe('RouteErrorPanel', () => {
  it('shows the title, error message, and a media icon', () => {
    renderWithRouterError({ title: 'Evaluator' }, new Error('Test error message'));

    expect(screen.getByTestId('error-panel')).toBeInTheDocument();
    expect(screen.getByText('Evaluator')).toBeInTheDocument();
    expect(screen.getByText("Couldn't Load Data")).toBeInTheDocument();
    expect(screen.getByText('Test error message')).toBeInTheDocument();
    expect(screen.getByTestId('nv-status-message-media')).toBeInTheDocument();
  });

  it('shows the API error detail', () => {
    renderWithRouterError({}, axiosError(500, 'Database unavailable'));

    expect(screen.getByText('Database unavailable')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('shows a not-found state without Retry for a 404', () => {
    renderWithRouterError({}, axiosError(404));

    expect(screen.getByText('Not Found')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Go Back' })).toBeInTheDocument();
  });

  it('shows a not-found state for a 404 route response', () => {
    renderWithRouterError({}, { status: 404, statusText: 'Not Found' });

    expect(screen.getByText('Not Found')).toBeInTheDocument();
  });

  it('shows a no-access state without Retry for a 403', () => {
    renderWithRouterError({}, axiosError(403));

    expect(screen.getByText('No Access')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
  });

  it('calls navigate(-1) when Go Back is clicked', async () => {
    const user = userEvent.setup();
    const navigateSpy = vi.fn();
    mockUseNavigate(navigateSpy);

    renderWithRouterError({}, new Error('Test error'));
    await user.click(screen.getByRole('button', { name: 'Go Back' }));

    expect(navigateSpy).toHaveBeenCalledWith(-1);
  });

  it('resets failed queries and re-renders the route on Retry', async () => {
    const user = userEvent.setup();
    const queryClient = new QueryClient();
    const resetSpy = vi.spyOn(queryClient, 'resetQueries');
    const { stopThrowing } = renderWithRouterError({}, new Error('Flaky'), queryClient);

    stopThrowing();
    await user.click(screen.getByRole('button', { name: 'Retry' }));

    expect(resetSpy).toHaveBeenCalledTimes(1);
    expect(await screen.findByText('Recovered')).toBeInTheDocument();
  });
});
