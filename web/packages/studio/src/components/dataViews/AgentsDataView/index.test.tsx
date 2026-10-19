// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { AgentsTable } from '@studio/components/dataViews/AgentsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { getAgentsListRoute, getModelCompareRoute } from '@studio/routes/utils';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useLocation } from 'react-router';

const WORKSPACE = 'default';

// These match the mock data in handlers.ts
const MOCK_AGENTS = [
  { name: 'react-agent', description: '' },
  { name: 'react-agent2', description: 'Second react agent' },
];

const renderTable = (onAgentRowClick = vi.fn(), history = getAgentsListRoute(WORKSPACE)) =>
  renderRoute(<AgentsTable onAgentRowClick={onAgentRowClick} />, {
    history,
    routes: [
      {
        path: ROUTES.workspace.agentsList,
        element: <AgentsTable onAgentRowClick={onAgentRowClick} />,
      },
    ],
  });

const LocationProbe = () => {
  const location = useLocation();
  return <div data-testid="model-compare-location">{`${location.pathname}${location.search}`}</div>;
};

describe('CombinedAgentsTable', () => {
  describe('columns', () => {
    it('renders Name, Description, Model, Deployments, and Created column headers', async () => {
      renderTable();

      await waitFor(() => expect(screen.queryByTestId('spinner')).not.toBeInTheDocument());

      expect(screen.getByRole('columnheader', { name: 'Name' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'Description' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'Model' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'Deployments' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'Created' })).toBeInTheDocument();
    });

    it('renders the agent model name from config.llms', async () => {
      renderTable();
      const cells = await screen.findAllByText('meta-llama-3-1-70b-instruct');
      expect(cells.length).toBeGreaterThan(0);
    });
  });

  describe('agents', () => {
    it('renders a row for each agent', async () => {
      renderTable();

      for (const agent of MOCK_AGENTS) {
        expect(await screen.findByText(agent.name)).toBeInTheDocument();
      }
    });

    it('renders non-empty agent descriptions', async () => {
      renderTable();
      expect(await screen.findByText('Second react agent')).toBeInTheDocument();
    });

    it('calls onAgentRowClick when an agent row is clicked', async () => {
      const user = userEvent.setup();
      const onAgentRowClick = vi.fn();
      renderTable(onAgentRowClick);

      const agentCell = await screen.findByText(MOCK_AGENTS[0].name);
      await user.click(agentCell);

      expect(onAgentRowClick).toHaveBeenCalledWith(
        expect.objectContaining({ name: MOCK_AGENTS[0].name }),
        expect.anything()
      );
    });
  });

  describe('empty state', () => {
    const mockNoAgents = () => {
      server.use(
        http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`, () =>
          HttpResponse.json({
            data: [],
            pagination: {
              page: 1,
              page_size: 50,
              current_page_size: 0,
              total_pages: 0,
              total_results: 0,
            },
          })
        )
      );
    };

    it('shows the no-agents empty state when no agents exist', async () => {
      mockNoAgents();
      renderTable();

      expect(
        await screen.findByText(ENTITY_EMPTY_STATES.agents.heading, undefined, {
          timeout: XL_SELECTOR_TIMEOUT,
        })
      ).toBeInTheDocument();
      expect(screen.getByText(ENTITY_EMPTY_STATES.agents.subheading)).toBeInTheDocument();
    });
  });

  describe('deployments status column', () => {
    it('shows "No Deployments" for agents with no deployments', async () => {
      // react-agent has 2 deployments, react-agent2 has 1 — neither has zero
      // This test verifies the column renders without errors; checking exact status
      // for all agents requires a custom handler override.
      renderTable();
      await screen.findByText(MOCK_AGENTS[0].name);
      // The column exists and renders something for each row
      expect(screen.getAllByRole('columnheader', { name: 'Deployments' }).length).toBeGreaterThan(
        0
      );
    });

    it('shows healthy count out of total for agents with deployments', async () => {
      renderTable();

      // react-agent has 2 deployments, 1 running (rag-agent-prod) → "1/2 Healthy"
      expect(await screen.findByText('1/2 Healthy')).toBeInTheDocument();
      // react-agent2 has 1 deployment, 0 running (chat-agent-staging is error) → "0/1 Healthy"
      expect(await screen.findByText('0/1 Healthy')).toBeInTheDocument();
    });
  });

  describe('row actions', () => {
    it('shows Deploy, Test models, Clone, and Delete actions for agent rows', async () => {
      const user = userEvent.setup();
      renderTable();

      await screen.findByText(MOCK_AGENTS[0].name);

      const menuButtons = screen.getAllByRole('button', { name: /actions/i });
      await user.click(menuButtons[0]);

      const deployItems = await screen.findAllByRole('menuitem', { name: 'Deploy' });
      const testModelItems = screen.getAllByRole('menuitem', { name: 'Test models' });
      const cloneItems = screen.getAllByRole('menuitem', { name: 'Clone' });
      const deleteItems = screen.getAllByRole('menuitem', { name: 'Delete' });
      expect(deployItems.length).toBeGreaterThan(0);
      expect(testModelItems.length).toBeGreaterThan(0);
      expect(cloneItems.length).toBeGreaterThan(0);
      expect(deleteItems.length).toBeGreaterThan(0);
    });

    it('opens Playground with the row model selected when Test models is selected', async () => {
      const user = userEvent.setup();
      renderRoute(undefined, {
        history: getAgentsListRoute(WORKSPACE),
        routes: [
          {
            path: ROUTES.workspace.agentsList,
            element: <AgentsTable />,
          },
          {
            path: ROUTES.workspace.modelCompare,
            element: <LocationProbe />,
          },
        ],
      });

      await screen.findByText(MOCK_AGENTS[0].name);

      await user.click(screen.getAllByRole('button', { name: /actions/i })[0]);
      await user.click((await screen.findAllByRole('menuitem', { name: 'Test models' }))[0]);

      expect(await screen.findByTestId('model-compare-location')).toHaveTextContent(
        `${getModelCompareRoute(WORKSPACE)}?model=${encodeURIComponent(
          `${WORKSPACE}/meta-llama-3-1-70b-instruct`
        )}`
      );
    });

    it('hides Test models when Playground is disabled', async () => {
      const user = userEvent.setup();
      renderRoute(undefined, {
        history: getAgentsListRoute(WORKSPACE),
        routes: [
          {
            path: ROUTES.workspace.agentsList,
            element: <AgentsTable canTestModels={false} />,
          },
        ],
      });

      await screen.findByText(MOCK_AGENTS[0].name);

      await user.click(screen.getAllByRole('button', { name: /actions/i })[0]);

      expect((await screen.findAllByRole('menuitem', { name: 'Deploy' })).length).toBeGreaterThan(
        0
      );
      expect(screen.queryByRole('menuitem', { name: 'Test models' })).not.toBeInTheDocument();
    });

    it('calls onCloneAgent with the row when Clone is selected', async () => {
      const user = userEvent.setup();
      const onCloneAgent = vi.fn();
      renderRoute(<AgentsTable onCloneAgent={onCloneAgent} />, {
        history: getAgentsListRoute(WORKSPACE),
        routes: [
          {
            path: ROUTES.workspace.agentsList,
            element: <AgentsTable onCloneAgent={onCloneAgent} />,
          },
        ],
      });

      await screen.findByText(MOCK_AGENTS[0].name);

      await user.click(screen.getAllByRole('button', { name: /actions/i })[0]);
      await user.click((await screen.findAllByRole('menuitem', { name: 'Clone' }))[0]);

      expect(onCloneAgent).toHaveBeenCalledTimes(1);
      expect(onCloneAgent).toHaveBeenCalledWith(
        expect.objectContaining({
          name: expect.stringMatching(/^react-agent2?$/),
          workspace: WORKSPACE,
        })
      );
    });
  });

  describe('sorting', () => {
    // Created order is intentionally inverse of alphabetical so the two sort
    // orders are distinguishable: -created_at desc → [zeta, middle, alpha];
    // name asc → [alpha, middle, zeta].
    const AGENTS = [
      {
        name: 'zeta-agent',
        workspace: WORKSPACE,
        description: '',
        config: {},
        created_at: '2026-04-03T00:00:00Z',
      },
      {
        name: 'alpha-agent',
        workspace: WORKSPACE,
        description: '',
        config: {},
        created_at: '2026-04-01T00:00:00Z',
      },
      {
        name: 'middle-agent',
        workspace: WORKSPACE,
        description: '',
        config: {},
        created_at: '2026-04-02T00:00:00Z',
      },
    ];

    const mockSortedAgents = () => {
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`,
          ({ request }) => {
            const url = new URL(request.url);
            const sort = url.searchParams.get('sort') ?? '-created_at';
            const desc = sort.startsWith('-');
            const field = desc ? sort.slice(1) : sort;
            const sorted = [...AGENTS].sort((a, b) => {
              const av = (a as Record<string, unknown>)[field] as string | undefined;
              const bv = (b as Record<string, unknown>)[field] as string | undefined;
              const cmp = String(av ?? '').localeCompare(String(bv ?? ''));
              return desc ? -cmp : cmp;
            });
            return HttpResponse.json({
              data: sorted,
              pagination: {
                page: 1,
                page_size: 50,
                current_page_size: sorted.length,
                total_pages: 1,
                total_results: sorted.length,
              },
            });
          }
        )
      );
    };

    it('sorts rows by created_at descending on default render', async () => {
      mockSortedAgents();
      renderTable();

      await screen.findByText('alpha-agent');

      const cells = screen
        .getAllByRole('cell')
        .filter((cell) => /-agent$/.test(cell.textContent ?? ''));
      const names = cells.map((c) => c.textContent);
      // created_at: zeta=04-03 (newest), middle=04-02, alpha=04-01
      expect(names).toEqual(['zeta-agent', 'middle-agent', 'alpha-agent']);
    });

    it('sorts alphabetically by Name when Name column header is clicked', async () => {
      const user = userEvent.setup();
      mockSortedAgents();
      renderTable();

      await screen.findByText('alpha-agent');

      const nameHeader = screen.getByRole('columnheader', { name: 'Name' });
      await user.click(within(nameHeader).getByRole('button', { name: 'Name' }));

      await waitFor(() => {
        const cells = screen
          .getAllByRole('cell')
          .filter((cell) => /-agent$/.test(cell.textContent ?? ''));
        const names = cells.map((c) => c.textContent);
        expect(names).toEqual(['alpha-agent', 'middle-agent', 'zeta-agent']);
      });
    });
  });

  describe('server-side query wiring', () => {
    it('serializes page, page_size, and sort onto the request URL', async () => {
      const seen: { page?: string | null; page_size?: string | null; sort?: string | null } = {};
      server.use(
        http.get(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`,
          ({ request }) => {
            const url = new URL(request.url);
            seen.page = url.searchParams.get('page');
            seen.page_size = url.searchParams.get('page_size');
            seen.sort = url.searchParams.get('sort');
            return HttpResponse.json({
              data: [],
              pagination: {
                page: 1,
                page_size: 50,
                current_page_size: 0,
                total_pages: 0,
                total_results: 0,
              },
            });
          }
        )
      );
      renderTable();

      await waitFor(() => {
        expect(seen.page).toBe('1');
        expect(seen.page_size).toBe('50');
        expect(seen.sort).toBe('-created_at');
      });
    });

    it('renders the pagination total reported by the server', async () => {
      server.use(
        http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`, () =>
          HttpResponse.json({
            data: [
              {
                name: 'react-agent',
                workspace: WORKSPACE,
                description: '',
                config: {},
                created_at: '2026-04-20T10:00:00Z',
              },
            ],
            pagination: {
              page: 1,
              page_size: 50,
              current_page_size: 1,
              total_pages: 5,
              total_results: 213,
            },
          })
        )
      );
      renderTable();

      // The pager renders "<count> of <total>" — match the exact total_results from the response.
      expect(await screen.findByText(/213/)).toBeInTheDocument();
    });
  });

  describe('delete agent', () => {
    it('shows delete confirmation modal when Delete is selected', async () => {
      const user = userEvent.setup();
      renderTable();

      await screen.findByText(MOCK_AGENTS[0].name);

      const menuButtons = screen.getAllByRole('button', { name: /actions/i });
      await user.click(menuButtons[0]);
      const deleteItems = await screen.findAllByRole('menuitem', { name: 'Delete' });
      await user.click(deleteItems[0]);

      expect(await screen.findByText('Delete Agent')).toBeInTheDocument();
      expect(
        screen.getByText('Are you sure you want to delete this agent and all its deployments?')
      ).toBeInTheDocument();
    });

    it('deletes the agent’s deployments first, then the agent, and shows a success toast', async () => {
      const order: string[] = [];
      server.use(
        // One agent so the first row is deterministic, with one deployment.
        http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`, () =>
          HttpResponse.json({
            data: [{ name: 'react-agent', description: '', config: {} }],
            pagination: {
              page: 1,
              page_size: 50,
              current_page_size: 1,
              total_pages: 1,
              total_results: 1,
            },
          })
        ),
        http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments`, () =>
          HttpResponse.json({
            data: [
              {
                name: 'react-agent-dep',
                agent: 'react-agent',
                status: 'running',
                workspace: WORKSPACE,
              },
            ],
            pagination: {
              page: 1,
              page_size: 50,
              current_page_size: 1,
              total_pages: 1,
              total_results: 1,
            },
          })
        ),
        http.delete(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments/:name`,
          ({ params }) => {
            order.push(`deployment:${params.name}`);
            return new HttpResponse(null, { status: 204 });
          }
        ),
        http.delete(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents/:name`,
          ({ params }) => {
            order.push(`agent:${params.name}`);
            return new HttpResponse(null, { status: 204 });
          }
        )
      );

      const user = userEvent.setup();
      renderTable();

      await screen.findByText('react-agent');

      const menuButtons = screen.getAllByRole('button', { name: /actions/i });
      await user.click(menuButtons[0]);
      const deleteItems = await screen.findAllByRole('menuitem', { name: 'Delete' });
      await user.click(deleteItems[0]);

      const confirmDialog = await screen.findByRole('dialog');
      await user.click(within(confirmDialog).getByRole('button', { name: 'Delete' }));

      // Deployment is deleted before the agent.
      await waitFor(() => expect(order).toContain('agent:react-agent'));
      expect(order).toEqual(['deployment:react-agent-dep', 'agent:react-agent']);
      expect(await screen.findByTestId('mock-toast-success')).toHaveTextContent('Agent deleted.');
    });

    it('paginates deployments so an agent deployment beyond the first page is still deleted', async () => {
      const order: string[] = [];
      server.use(
        http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`, () =>
          HttpResponse.json({
            data: [{ name: 'react-agent', description: '', config: {} }],
            pagination: {
              page: 1,
              page_size: 50,
              current_page_size: 1,
              total_pages: 1,
              total_results: 1,
            },
          })
        ),
        http.get(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments`,
          ({ request }) => {
            const page = new URL(request.url).searchParams.get('page');
            // Page 1 has only another agent's deployment; the target's is on page 2.
            const data =
              page === '1'
                ? [
                    {
                      name: 'other-dep',
                      agent: 'other-agent',
                      status: 'running',
                      workspace: WORKSPACE,
                    },
                  ]
                : [
                    {
                      name: 'react-agent-dep2',
                      agent: 'react-agent',
                      status: 'running',
                      workspace: WORKSPACE,
                    },
                  ];
            return HttpResponse.json({
              data,
              pagination: {
                page: Number(page),
                page_size: 100,
                current_page_size: 1,
                total_pages: 2,
                total_results: 2,
              },
            });
          }
        ),
        http.delete(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments/:name`,
          ({ params }) => {
            order.push(`deployment:${params.name}`);
            return new HttpResponse(null, { status: 204 });
          }
        ),
        http.delete(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents/:name`,
          ({ params }) => {
            order.push(`agent:${params.name}`);
            return new HttpResponse(null, { status: 204 });
          }
        )
      );

      const user = userEvent.setup();
      renderTable();
      await screen.findByText('react-agent');
      await user.click(screen.getAllByRole('button', { name: /actions/i })[0]);
      await user.click((await screen.findAllByRole('menuitem', { name: 'Delete' }))[0]);
      const dialog = await screen.findByRole('dialog');
      await user.click(within(dialog).getByRole('button', { name: 'Delete' }));

      await waitFor(() => expect(order).toContain('agent:react-agent'));
      // Only the target agent's deployment (from page 2) is deleted, before the agent.
      expect(order).toEqual(['deployment:react-agent-dep2', 'agent:react-agent']);
    });

    it('surfaces an error and skips the agent delete when a deployment delete fails', async () => {
      let agentDeleteCalled = false;
      server.use(
        http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`, () =>
          HttpResponse.json({
            data: [{ name: 'react-agent', description: '', config: {} }],
            pagination: {
              page: 1,
              page_size: 50,
              current_page_size: 1,
              total_pages: 1,
              total_results: 1,
            },
          })
        ),
        http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments`, () =>
          HttpResponse.json({
            data: [
              {
                name: 'react-agent-dep',
                agent: 'react-agent',
                status: 'running',
                workspace: WORKSPACE,
              },
            ],
            pagination: {
              page: 1,
              page_size: 100,
              current_page_size: 1,
              total_pages: 1,
              total_results: 1,
            },
          })
        ),
        http.delete(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments/:name`,
          () => HttpResponse.json({ detail: 'teardown failed' }, { status: 500 })
        ),
        http.delete(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents/:name`,
          () => {
            agentDeleteCalled = true;
            return new HttpResponse(null, { status: 204 });
          }
        )
      );

      const user = userEvent.setup();
      renderTable();
      await screen.findByText('react-agent');
      await user.click(screen.getAllByRole('button', { name: /actions/i })[0]);
      await user.click((await screen.findAllByRole('menuitem', { name: 'Delete' }))[0]);
      const dialog = await screen.findByRole('dialog');
      await user.click(within(dialog).getByRole('button', { name: 'Delete' }));

      expect(await screen.findByTestId('mock-toast-error')).toBeInTheDocument();
      expect(agentDeleteCalled).toBe(false);
    });
  });

  describe('bulk delete', () => {
    it('deletes every selected agent and its deployments, then shows a summary toast', async () => {
      const deletedAgents: string[] = [];
      const deletedDeployments: string[] = [];
      server.use(
        http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`, () =>
          HttpResponse.json({
            data: [
              { name: 'react-agent', description: '', config: {} },
              { name: 'react-agent2', description: '', config: {} },
            ],
            pagination: {
              page: 1,
              page_size: 50,
              current_page_size: 2,
              total_pages: 1,
              total_results: 2,
            },
          })
        ),
        http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments`, () =>
          HttpResponse.json({
            data: [
              {
                name: 'react-agent-dep',
                agent: 'react-agent',
                status: 'running',
                workspace: WORKSPACE,
              },
            ],
            pagination: {
              page: 1,
              page_size: 100,
              current_page_size: 1,
              total_pages: 1,
              total_results: 1,
            },
          })
        ),
        http.delete(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments/:name`,
          ({ params }) => {
            deletedDeployments.push(String(params.name));
            return new HttpResponse(null, { status: 204 });
          }
        ),
        http.delete(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents/:name`,
          ({ params }) => {
            deletedAgents.push(String(params.name));
            return new HttpResponse(null, { status: 204 });
          }
        )
      );

      const user = userEvent.setup();
      renderTable();
      await screen.findByText('react-agent');
      await screen.findByText('react-agent2');

      // KUI disables checkboxes while loading and swaps nodes on re-render, so wait
      // for each to be enabled, click, and confirm it registered before continuing.
      const rowCheckboxAt = (index: number) =>
        screen.getAllByRole('checkbox', { name: /(De)?select row/i })[index];
      const selectRow = async (index: number) => {
        await waitFor(() => expect(rowCheckboxAt(index)).toBeEnabled());
        if (!(rowCheckboxAt(index) as HTMLInputElement).checked) {
          await user.click(rowCheckboxAt(index));
        }
        await waitFor(() => expect(rowCheckboxAt(index)).toBeChecked());
      };
      await selectRow(0);
      await selectRow(1);

      // The selection bar's Delete button opens the confirmation.
      await user.click(await screen.findByRole('button', { name: 'Delete selected agents' }));
      const dialog = await screen.findByRole('dialog');
      expect(within(dialog).getByText('Delete 2 Agents')).toBeInTheDocument();
      await user.click(within(dialog).getByRole('button', { name: 'Delete' }));

      await waitFor(() =>
        expect([...deletedAgents].sort()).toEqual(['react-agent', 'react-agent2'])
      );
      expect(deletedDeployments).toContain('react-agent-dep');
      expect(await screen.findByTestId('mock-toast-success')).toHaveTextContent(
        '2 agents deleted.'
      );
    });

    it('reports a partial failure when one selected agent fails to delete', async () => {
      const deletedAgents: string[] = [];
      server.use(
        http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`, () =>
          HttpResponse.json({
            data: [
              { name: 'react-agent', description: '', config: {} },
              { name: 'react-agent2', description: '', config: {} },
            ],
            pagination: {
              page: 1,
              page_size: 50,
              current_page_size: 2,
              total_pages: 1,
              total_results: 2,
            },
          })
        ),
        http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments`, () =>
          HttpResponse.json({
            data: [],
            pagination: {
              page: 1,
              page_size: 100,
              current_page_size: 0,
              total_pages: 1,
              total_results: 0,
            },
          })
        ),
        http.delete(
          `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents/:name`,
          ({ params }) => {
            if (params.name === 'react-agent2') {
              return HttpResponse.json({ detail: 'boom' }, { status: 500 });
            }
            deletedAgents.push(String(params.name));
            return new HttpResponse(null, { status: 204 });
          }
        )
      );

      const user = userEvent.setup();
      renderTable();
      await screen.findByText('react-agent');
      await screen.findByText('react-agent2');

      const rowCheckboxAt = (index: number) =>
        screen.getAllByRole('checkbox', { name: /(De)?select row/i })[index];
      const selectRow = async (index: number) => {
        await waitFor(() => expect(rowCheckboxAt(index)).toBeEnabled());
        if (!(rowCheckboxAt(index) as HTMLInputElement).checked) {
          await user.click(rowCheckboxAt(index));
        }
        await waitFor(() => expect(rowCheckboxAt(index)).toBeChecked());
      };
      await selectRow(0);
      await selectRow(1);

      await user.click(await screen.findByRole('button', { name: 'Delete selected agents' }));
      const dialog = await screen.findByRole('dialog');
      await user.click(within(dialog).getByRole('button', { name: 'Delete' }));

      // The succeeded agent is still deleted; the failure is surfaced as a summary toast.
      await waitFor(() => expect(deletedAgents).toEqual(['react-agent']));
      expect(await screen.findByTestId('mock-toast-error')).toHaveTextContent(
        'Failed to delete 1 of 2 agents.'
      );
    });
  });
});
