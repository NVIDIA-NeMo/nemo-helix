// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getListSpansQueryKey } from '@nemo/sdk/generated/platform/spans';
import { IntakeSpansTable } from '@studio/components/IntakeLists/IntakeSpansTable';
import { ROUTES } from '@studio/constants/routes';
import { mockSpansPage } from '@studio/mocks/intake/telemetry';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

describe('IntakeSpansTable', () => {
  it('shows bounded input and output summaries', async () => {
    const requestedModes: Array<string | null> = [];
    server.use(
      http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), ({ request }) => {
        requestedModes.push(new URL(request.url).searchParams.get('mode'));
        return HttpResponse.json(mockSpansPage);
      })
    );

    renderRoute(<IntakeSpansTable workspace="default" />, {
      history: '/workspaces/default/intake/spans',
    });

    expect(
      await screen.findByText('Can I deploy this model in a private workspace?')
    ).toBeInTheDocument();
    expect(
      screen.getAllByText(
        'Yes. Use a private workspace and restrict access through workspace membership.'
      ).length
    ).toBeGreaterThan(0);
    expect(requestedModes).toContain('preview');
  });

  it('opens span rows in the session detail route with trace and span deep links', async () => {
    const user = userEvent.setup();

    renderRoute(undefined, {
      history: '/workspaces/default/intake/spans',
      routes: [
        {
          path: ROUTES.workspace.intakeSpans,
          element: <IntakeSpansTable workspace="default" />,
        },
        {
          path: ROUTES.workspace.intakeSession,
          element: <LocationDisplay />,
        },
      ],
    });

    await user.click(await screen.findByText('Answer customer policy question'));

    expect(await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(
      '/workspaces/default/intake/sessions/session-agent-run-001?traceId=trace-agent-run-001&spanId=span-root-001'
    );
  });

  it('seeds a clearable 30-day started_at filter into span list requests', async () => {
    const user = userEvent.setup();
    const startedAtParams: Array<string | null> = [];
    server.use(
      http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), ({ request }) => {
        startedAtParams.push(new URL(request.url).searchParams.get('filter[started_at][$gte]'));
        return HttpResponse.json(mockSpansPage);
      })
    );

    renderRoute(<IntakeSpansTable workspace="default" />, {
      history: '/workspaces/default/intake/spans',
    });

    await screen.findByText('Answer customer policy question');
    await waitFor(() => expect(startedAtParams.filter(Boolean).length).toBeGreaterThan(0));

    const seededGte = new Date(startedAtParams.filter(Boolean).at(-1) as string);
    const daysAgo = (Date.now() - seededGte.getTime()) / 86_400_000;
    expect(daysAgo).toBeGreaterThanOrEqual(29);
    expect(daysAgo).toBeLessThanOrEqual(31);

    await user.click(screen.getByTestId('clear-filters'));
    await waitFor(() => expect(startedAtParams.at(-1)).toBeNull());
  });

  it('shows explicit span filter facets', async () => {
    const user = userEvent.setup();

    renderRoute(<IntakeSpansTable workspace="default" />, {
      history: '/workspaces/default/intake/spans',
    });

    await screen.findByText('Answer customer policy question');
    await user.click(await screen.findByTestId('open-filters-button'));

    expect(await screen.findByText('Trace ID')).toBeInTheDocument();
    expect(screen.getAllByText('Kind').length).toBeGreaterThan(1);
    expect(screen.getAllByText('Status').length).toBeGreaterThan(1);
    expect(screen.getByText('Started At')).toBeInTheDocument();
    expect(screen.queryByText('Session ID')).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Search by span ID')).not.toBeInTheDocument();
  });
});
