// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { getListTracesQueryKey } from '@nemo/sdk/generated/platform/traces';
import { IntakeTracesTable } from '@studio/components/IntakeLists/IntakeTracesTable';
import { ROUTES } from '@studio/constants/routes';
import { mockTracesPage } from '@studio/mocks/intake/telemetry';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

describe('IntakeTracesTable', () => {
  it('loads trace rows in preview mode for bounded payloads and aggregate metrics', async () => {
    const requestedModes: Array<string | null> = [];
    server.use(
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), ({ request }) => {
        requestedModes.push(new URL(request.url).searchParams.get('mode'));
        return HttpResponse.json(mockTracesPage);
      })
    );

    renderRoute(<IntakeTracesTable workspace="default" />, {
      history: '/workspaces/default/intake/traces',
    });

    await screen.findByText('Answer customer policy question');
    expect(screen.getAllByText('session-agent-run-001')).not.toHaveLength(0);
    expect(screen.getByText('Can I deploy this model in a private workspace?')).toBeInTheDocument();
    expect(
      screen.getByText(
        'Yes. Use a private workspace and restrict access through workspace membership.'
      )
    ).toBeInTheDocument();

    await waitFor(() => expect(requestedModes).toContain('preview'));
    expect(requestedModes).not.toContain('detailed');
    expect(requestedModes).not.toContain('summary');
  });

  it('opens trace rows in the canonical session detail route', async () => {
    const user = userEvent.setup();

    renderRoute(undefined, {
      history: '/workspaces/default/intake/traces',
      routes: [
        {
          path: ROUTES.workspace.intakeTraces,
          element: <IntakeTracesTable workspace="default" />,
        },
        {
          path: ROUTES.workspace.intakeSession,
          element: <LocationDisplay />,
        },
      ],
    });

    await user.click(await screen.findByText('Answer customer policy question'));

    expect(await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(
      '/workspaces/default/intake/sessions/session-agent-run-001?traceId=trace-agent-run-001'
    );
  });

  it('seeds a clearable 30-day started_at filter into trace list requests', async () => {
    const user = userEvent.setup();
    const startedAtParams: Array<string | null> = [];
    server.use(
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), ({ request }) => {
        startedAtParams.push(new URL(request.url).searchParams.get('filter[started_at][$gte]'));
        return HttpResponse.json(mockTracesPage);
      })
    );

    renderRoute(<IntakeTracesTable workspace="default" />, {
      history: '/workspaces/default/intake/traces',
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

  it('shows explicit trace filter facets', async () => {
    const user = userEvent.setup();

    renderRoute(<IntakeTracesTable workspace="default" />, {
      history: '/workspaces/default/intake/traces',
    });

    await screen.findByText('Answer customer policy question');
    await user.click(await screen.findByTestId('open-filters-button'));

    expect(await screen.findByText('Trace ID')).toBeInTheDocument();
    expect(screen.getByText('Started At')).toBeInTheDocument();
    expect(screen.getAllByText('Status')).not.toHaveLength(0);
    expect(screen.getByText('Session ID')).toBeInTheDocument();
    expect(screen.queryByText('Evaluation Run ID')).not.toBeInTheDocument();
  });
  it('offers the intake skill and CLI when no traces have been ingested', async () => {
    const user = userEvent.setup();
    server.use(
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), () =>
        HttpResponse.json({ data: [], pagination: { total_results: 0 } })
      )
    );

    renderRoute(<IntakeTracesTable workspace="test-workspace" />, {
      history: '/workspaces/default/intake/traces',
    });

    const descriptor = ENTITY_EMPTY_STATES.telemetryTraces;
    expect(await screen.findByText(descriptor.heading)).toBeInTheDocument();

    const help = await screen.findByTestId('entity-empty-state-help');
    expect(help).toHaveTextContent(
      descriptor.skillPrompt!.replaceAll('<workspace>', 'test-workspace')
    );
    expect(descriptor.skillPrompt).toMatch(/nemo-intake skill/);
    expect(help).not.toHaveTextContent('<workspace>');

    await user.click(screen.getByRole('radio', { name: 'CLI' }));
    expect(help).toHaveTextContent(
      descriptor.cliCommand!.replaceAll('<workspace>', 'test-workspace')
    );
    expect(help).not.toHaveTextContent('<workspace>');
  });
});
