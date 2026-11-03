// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { JOB_POLLING_INTERVAL_LONG } from '@nemo/common/src/constants';
import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { getListTracesQueryKey } from '@nemo/sdk/generated/platform/traces';
import { IntakeTracesTable } from '@studio/components/IntakeLists/IntakeTracesTable';
import { ROUTES } from '@studio/constants/routes';
import { mockTracesPage } from '@studio/mocks/intake/telemetry';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { getAgentDetailRoute, getIntakeTracesRoute } from '@studio/routes/utils';
import { LG_SELECTOR_TIMEOUT, LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { act, renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { within } from '@testing-library/react';
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

  it('links registered trace agents to the agent page and leaves unknown names as text', async () => {
    const [registered, unregistered] = mockTracesPage.data;
    server.use(
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), () =>
        HttpResponse.json({
          ...mockTracesPage,
          data: [
            { ...registered, agent_name: 'react-agent' },
            { ...unregistered, agent_name: 'unregistered-agent' },
          ],
        })
      )
    );

    renderRoute(<IntakeTracesTable workspace="default" />, {
      history: '/workspaces/default/intake/traces',
    });

    expect(
      await screen.findByRole('link', { name: 'react-agent' }, { timeout: LG_SELECTOR_TIMEOUT })
    ).toHaveAttribute('href', getAgentDetailRoute('default', 'react-agent'));
    expect(screen.getByText('unregistered-agent')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'unregistered-agent' })).not.toBeInTheDocument();
  });

  it('opens the agent page from the agent link instead of the trace row', async () => {
    const user = userEvent.setup();
    server.use(
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), () =>
        HttpResponse.json({
          ...mockTracesPage,
          data: [{ ...mockTracesPage.data[0], agent_name: 'react-agent' }],
        })
      )
    );

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
        {
          path: ROUTES.workspace.agentDetail,
          element: <LocationDisplay />,
        },
      ],
    });

    await user.click(
      await screen.findByRole('link', { name: 'react-agent' }, { timeout: LG_SELECTOR_TIMEOUT })
    );

    expect(await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(
      getAgentDetailRoute('default', 'react-agent')
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

  it('does not update the trace URL or refetch a seeded Started At range before Apply', async () => {
    vi.stubEnv('TZ', 'America/Los_Angeles');
    vi.setSystemTime(new Date('2026-05-15T12:00:00Z'));
    try {
      const user = userEvent.setup();
      const requests: string[] = [];
      server.use(
        http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), ({ request }) => {
          requests.push(request.url);
          return HttpResponse.json(mockTracesPage);
        })
      );
      renderRoute(
        <>
          <IntakeTracesTable workspace={DEFAULT_WORKSPACE} />
          <LocationDisplay />
        </>,
        { history: getIntakeTracesRoute(DEFAULT_WORKSPACE) }
      );
      await screen.findByText('Answer customer policy question');
      // Restore the clock after seeding; use-debounce relies on Date.now advancing.
      vi.useRealTimers();
      await user.click(screen.getByTestId('open-filters-button'));
      const originalUrl = screen.getByTestId(LOCATION_DISPLAY_TEST_ID).textContent;
      const originalRequests = [...requests];
      const input = within(screen.getByTestId('column-filter-started_at')).getAllByRole(
        'textbox'
      )[0];
      await user.click(input);
      await user.click(await screen.findByRole('button', { name: /April 20th, 2026/ }));
      expect(screen.getByTestId(LOCATION_DISPLAY_TEST_ID).textContent).toBe(originalUrl);
      expect(requests).toEqual(originalRequests);
      await user.click(screen.getByRole('button', { name: 'Cancel' }));
      expect(screen.getByTestId(LOCATION_DISPLAY_TEST_ID).textContent).toBe(originalUrl);
      expect(requests).toEqual(originalRequests);
      await user.click(screen.getByRole('button', { name: /April 20th, 2026/ }));
      await user.click(screen.getByRole('button', { name: 'Apply' }));
      await waitFor(() => expect(requests.length).toBeGreaterThan(originalRequests.length));
      expect(screen.getByTestId(LOCATION_DISPLAY_TEST_ID).textContent).not.toBe(originalUrl);
      expect(new URL(requests.at(-1)!).searchParams.get('filter[started_at][$lte]')).toBe(
        '2026-04-20T23:59:59.999Z'
      );
    } finally {
      vi.unstubAllEnvs();
      vi.useRealTimers();
    }
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
    expect(screen.getAllByText('Agent')).not.toHaveLength(0);
    expect(screen.queryByText('Evaluation Run ID')).not.toBeInTheDocument();
  });

  it('filters trace list requests by agent from a deep link', async () => {
    const agentParams: (string | null)[] = [];
    server.use(
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), ({ request }) => {
        agentParams.push(new URL(request.url).searchParams.get('filter[agent_name]'));
        return HttpResponse.json(mockTracesPage);
      })
    );

    renderRoute(<IntakeTracesTable workspace="default" />, {
      history: getIntakeTracesRoute('default', { agentName: 'email-security-triage' }),
    });

    await waitFor(() => expect(agentParams).toContain('email-security-triage'));
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
    // CodeSnippet updates its highlighted content asynchronously after the toggle.
    await waitFor(() =>
      expect(help).toHaveTextContent(
        descriptor.cliCommand!.replaceAll('<workspace>', 'test-workspace')
      )
    );
    expect(help).not.toHaveTextContent('<workspace>');
  });

  describe('polling', () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    it('shows newly ingested traces without a reload', async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      const [newTrace, existingTrace] = mockTracesPage.data;
      let traces = [existingTrace];
      server.use(
        http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), () =>
          HttpResponse.json({
            ...mockTracesPage,
            data: traces,
            pagination: { ...mockTracesPage.pagination, total_results: traces.length },
          })
        )
      );

      renderRoute(<IntakeTracesTable workspace="default" />, {
        history: '/workspaces/default/intake/traces',
      });
      expect(await screen.findByText('Answer customer policy question')).toBeInTheDocument();
      expect(
        screen.queryByText('Explain private workspace access controls')
      ).not.toBeInTheDocument();

      traces = [newTrace, existingTrace];
      await act(() => vi.advanceTimersByTimeAsync(JOB_POLLING_INTERVAL_LONG));

      expect(
        await screen.findByText('Explain private workspace access controls')
      ).toBeInTheDocument();
    });

    it('keeps the current rows visible while a poll is in flight', async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      let requestCount = 0;
      let releasePoll: () => void = () => {};
      const pollHeld = new Promise<void>((resolve) => {
        releasePoll = resolve;
      });
      server.use(
        http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), async () => {
          requestCount += 1;
          if (requestCount > 1) await pollHeld;
          return HttpResponse.json(mockTracesPage);
        })
      );

      renderRoute(<IntakeTracesTable workspace="default" />, {
        history: '/workspaces/default/intake/traces',
      });
      expect(await screen.findByText('Answer customer policy question')).toBeInTheDocument();

      await act(() => vi.advanceTimersByTimeAsync(JOB_POLLING_INTERVAL_LONG));
      await waitFor(() => expect(requestCount).toBe(2));

      expect(screen.getByText('Answer customer policy question')).toBeInTheDocument();

      releasePoll();
    });

    it('keeps the current rows visible when a poll fails', async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      let requestCount = 0;
      server.use(
        http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), () => {
          requestCount += 1;
          if (requestCount > 1) {
            return HttpResponse.json({ detail: 'Service unavailable' }, { status: 503 });
          }
          return HttpResponse.json(mockTracesPage);
        })
      );

      renderRoute(<IntakeTracesTable workspace="default" />, {
        history: '/workspaces/default/intake/traces',
      });
      expect(await screen.findByText('Answer customer policy question')).toBeInTheDocument();

      await act(() => vi.advanceTimersByTimeAsync(JOB_POLLING_INTERVAL_LONG));
      await waitFor(() => expect(requestCount).toBeGreaterThan(1));
      await act(() => vi.advanceTimersByTimeAsync(0));

      expect(screen.getByText('Answer customer policy question')).toBeInTheDocument();
      expect(screen.queryByText(/Service unavailable/)).not.toBeInTheDocument();
    });
  });
});
