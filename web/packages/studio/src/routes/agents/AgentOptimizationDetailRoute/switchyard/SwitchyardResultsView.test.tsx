// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { AgentOptimizationDetailRoute } from '@studio/routes/agents/AgentOptimizationDetailRoute';
import { getAgentOptimizationDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const workspace = workspace1.workspace;
const jobName = 'route-support-bot';
const OPTIMIZE_JOB_URL = `${PLATFORM_BASE_URL}/apis/agent-optimization/v2/workspaces/:workspace/jobs/run-strategy/:name`;
const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;
const AGENTS_URL = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`;

const VM_RANDOM = 'support-bot-1a2b3c4d';
const VM_STAGE = 'support-bot-9f8e7d6c';

const INDEX = {
  strategy: 'switchyard',
  agent: `${workspace}/support-bot`,
  workspace,
  combinations: [
    {
      virtual_model: VM_RANDOM,
      config_type: 'random_routing',
      capable: `${workspace}/frontier`,
      efficient: `${workspace}/nano`,
      judge: null,
      config: { strong_probability: 0.8 },
      agent_config: `agent-${VM_RANDOM}.yaml`,
    },
    {
      virtual_model: VM_STAGE,
      config_type: 'stage_router',
      capable: `${workspace}/frontier`,
      efficient: `${workspace}/nano`,
      judge: null,
      config: { confidence_threshold: 0.5 },
      agent_config: `agent-${VM_STAGE}.yaml`,
    },
  ],
};

const ROUTED_YAML = `config_format: nemo-agents-spec-v1
name: support-bot
models:
  default:
    provider: openai
    model: ${workspace}/${VM_RANDOM}
    api_key_env: NEMO_AGENTS_IGW_API_KEY
telemetry:
  agent_name: support-bot
`;

interface CapturedAgent {
  name?: string;
  config?: { name?: string; models?: { default?: { model?: string } } };
}

const mockSwitchyardRun = (status: HelixJobStatus = 'completed') => {
  const captured: { agent: CapturedAgent } = { agent: {} };
  server.use(
    http.get(OPTIMIZE_JOB_URL, () =>
      HttpResponse.json({
        name: jobName,
        workspace,
        status,
        created_at: '2026-10-01T09:00:00Z',
        updated_at: '2026-10-01T09:05:00Z',
        spec: {
          strategy: 'switchyard',
          agent: 'support-bot',
          models: ['frontier', 'nano'],
          routing_strategies: ['random_routing', 'stage_router'],
        },
      })
    ),
    http.get(`${OPTIMIZE_JOB_URL}/status`, () =>
      HttpResponse.json({
        id: 'opt-1',
        name: jobName,
        status,
        status_details: {},
        error_details: {},
        steps: [],
        created_at: '2026-10-01T09:00:00Z',
        updated_at: '2026-10-01T09:05:00Z',
      })
    ),
    http.get(`${OPTIMIZE_JOB_URL}/results`, () =>
      HttpResponse.json({
        data: [
          {
            name: 'switchyard',
            artifact_storage_type: 'fileset',
            artifact_url: `fileset://${workspace}/routing-artifacts#switchyard`,
          },
        ],
      })
    ),
    http.get(`${FILESETS_URL}/routing-artifacts/files`, () =>
      HttpResponse.json({
        data: [
          { path: 'switchyard/switchyard-result.json', size: 512, file_ref: 'index' },
          { path: `switchyard/agent-${VM_RANDOM}.yaml`, size: 128, file_ref: 'random' },
        ],
      })
    ),
    http.get(`${FILESETS_URL}/routing-artifacts/-/:path`, ({ params }) =>
      decodeURIComponent(String(params['path'])).endsWith('switchyard-result.json')
        ? HttpResponse.json(INDEX)
        : new HttpResponse(ROUTED_YAML)
    ),
    http.get(`${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/providers`, () =>
      HttpResponse.json({
        data: [
          {
            name: 'build',
            workspace,
            served_models: [
              { model_entity_id: `${workspace}/frontier`, served_model_name: 'meta/llama-70b' },
              { model_entity_id: `${workspace}/nano`, served_model_name: 'nvidia/nano-9b' },
            ],
          },
        ],
        pagination: { page: 1, page_size: 100, current_page_size: 1, total_pages: 1 },
      })
    ),
    http.get(`${AGENTS_URL}/:name`, ({ params }) =>
      HttpResponse.json({
        id: 'agent-support-bot',
        name: params['name'],
        workspace,
        description: 'Support bot',
        config: {},
        config_format: 'nemo-agents-spec-v1',
        created_at: '2026-09-01T00:00:00Z',
        updated_at: '2026-09-01T00:00:00Z',
      })
    ),
    http.get(`${FILESETS_URL}/:name`, () => new HttpResponse(null, { status: 404 })),
    http.post(AGENTS_URL, async ({ request }) => {
      captured.agent = (await request.json()) as CapturedAgent;
      return HttpResponse.json({ ...captured.agent, workspace });
    })
  );
  return captured;
};

const renderRun = () =>
  renderRoute(undefined, {
    history: getAgentOptimizationDetailRoute(workspace, jobName),
    routes: [
      { path: ROUTES.workspace.agentOptimizationDetail, element: <AgentOptimizationDetailRoute /> },
      { path: ROUTES.workspace.agentDetail, element: <div>Agent detail page</div> },
    ],
  });

describe('AgentOptimizationDetailRoute — switchyard', () => {
  it('shows the models and routing strategies while the run is in progress', async () => {
    mockSwitchyardRun('active');
    renderRun();

    const inProgress = await screen.findByTestId('study-in-progress');
    expect(within(inProgress).getByText('frontier, nano')).toBeInTheDocument();
    expect(within(inProgress).getByText('Random routing, Stage router')).toBeInTheDocument();
    expect(
      within(inProgress).getByText('Virtual models appear once the study finishes.')
    ).toBeInTheDocument();
  });

  it('lists every virtual model the run created', async () => {
    mockSwitchyardRun();
    renderRun();

    expect(await screen.findByRole('link', { name: VM_STAGE })).toBeInTheDocument();
    expect(screen.getByText('Model pairs')).toBeInTheDocument();
    expect(screen.getAllByText('Capable model share 0.8').length).toBeGreaterThan(0);
    expect(screen.getByText('Confidence threshold 0.5')).toBeInTheDocument();
  });

  it('starts the tested virtual model with an empty split and explains its strategy', async () => {
    mockSwitchyardRun();
    renderRun();

    const panel = await screen.findByTestId('routing-test-panel');
    const split = within(panel).getByRole('list', { name: 'Responses by model' });
    expect(within(split).getByRole('listitem', { name: 'Capable model' })).toHaveTextContent(
      '0 · 0%'
    );
    expect(within(split).getByRole('listitem', { name: 'Efficient model' })).toHaveTextContent(
      '0 · 0%'
    );
    expect(within(panel).getByRole('button', { name: 'Clear' })).toBeDisabled();
    expect(within(panel).getByText(/goes to the capable model with probability 80%/)).toBeVisible();
  });

  it('switches the tested virtual model when a row is clicked', async () => {
    const user = userEvent.setup();
    const scrollIntoView = vi.fn();
    window.HTMLElement.prototype.scrollIntoView = scrollIntoView;
    mockSwitchyardRun();
    renderRun();

    await user.click(await screen.findByText('Confidence threshold 0.5'));

    const panel = screen.getByTestId('routing-test-panel');
    expect(within(panel).getByRole('link', { name: VM_STAGE })).toBeInTheDocument();
    expect(scrollIntoView).toHaveBeenCalled();
  });

  it('deploys a route as a new agent pointed at its virtual model', async () => {
    const user = userEvent.setup();
    const captured = mockSwitchyardRun();
    renderRun();

    await user.click(await screen.findByRole('button', { name: `Deploy ${VM_RANDOM}` }));
    const dialog = await screen.findByRole('dialog', { name: 'Deploy route as a new agent' });
    expect(within(dialog).getByRole('textbox')).toHaveValue(VM_RANDOM);
    const deploy = await within(dialog).findByRole('button', { name: 'Deploy' });
    await waitFor(() => expect(deploy).toBeEnabled());
    await user.click(deploy);

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    expect(captured.agent.name).toBe(VM_RANDOM);
    expect(captured.agent.config?.name).toBe(VM_RANDOM);
    expect(captured.agent.config?.models?.default?.model).toBe(`${workspace}/${VM_RANDOM}`);
  });
});
