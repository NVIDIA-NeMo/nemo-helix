// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { DeployTrialModal } from '@studio/routes/agents/AgentOptimizationDetailRoute/DeployTrialModal';
import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import { getAgentOptimizationDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const workspace = workspace1.workspace;
const AGENTS_URL = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents`;
const FILE_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name/-/:path`;

const SPEC = {
  strategy: 'legacy',
  agent: 'hermes',
  optimize_config: 'optimize.yaml',
  optimize_config_fileset: `${workspace}/hermes-bundle`,
};

const OPTIMIZE_YAML = `
optimizer:
  search_space:
    temperature: {type: fabric, path: models.default.temperature, values: [0.0, 0.2]}
`;

const SOURCE_CONFIG = {
  config_format: 'nemo-agents-spec-v1',
  name: 'hermes',
  default_harness: 'main',
  harnesses: { main: { kind: 'hermes' } },
  models: { default: { provider: 'nvidia', model: 'llama', temperature: 0.7 } },
};

const TRIAL: Trial = {
  number: 4,
  state: 'COMPLETE',
  durationSeconds: 12,
  paretoOptimal: true,
  metrics: [],
  params: [{ name: 'temperature', value: '0.2' }],
};

interface CapturedAgent {
  name?: string;
  config_format?: string;
  config?: { name?: string; models?: { default?: { temperature?: number } } };
}

const mockHelix = (optimizeYaml = OPTIMIZE_YAML): { body: CapturedAgent } => {
  const captured: { body: CapturedAgent } = { body: {} };
  server.use(
    http.get(`${AGENTS_URL}/:name`, ({ params }) =>
      HttpResponse.json({
        id: 'agent-hermes',
        name: params['name'],
        workspace,
        description: 'Hermes',
        config: SOURCE_CONFIG,
        config_format: 'nemo-agents-spec-v1',
        created_at: '2026-04-01T00:00:00Z',
      })
    ),
    http.get(FILE_URL, () => new HttpResponse(optimizeYaml)),
    http.post(AGENTS_URL, async ({ request }) => {
      captured.body = (await request.json()) as CapturedAgent;
      return HttpResponse.json({ ...captured.body, workspace });
    })
  );
  return captured;
};

const renderModal = (trial: Trial = TRIAL) =>
  renderRoute(undefined, {
    history: getAgentOptimizationDetailRoute(workspace, 'study'),
    routes: [
      {
        path: ROUTES.workspace.agentOptimizationDetail,
        element: (
          <DeployTrialModal workspace={workspace} spec={SPEC} trial={trial} onClose={vi.fn()} />
        ),
      },
      { path: ROUTES.workspace.agentDetail, element: <div>Agent detail page</div> },
    ],
  });

describe('DeployTrialModal', () => {
  it('creates an agent with the trial configuration and opens it', async () => {
    const user = userEvent.setup();
    const captured = mockHelix();
    renderModal();

    const dialog = await screen.findByRole('dialog');
    expect(within(dialog).getByRole('textbox')).toHaveValue('hermes-trial-4');
    await within(dialog).findByText('models.default.temperature');

    const deploy = within(dialog).getByRole('button', { name: 'Deploy' });
    await waitFor(() => expect(deploy).toBeEnabled());
    await user.click(deploy);

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    expect(captured.body.name).toBe('hermes-trial-4');
    expect(captured.body.config_format).toBe('nemo-agents-spec-v1');
    expect(captured.body.config?.name).toBe('hermes-trial-4');
    expect(captured.body.config?.models?.default?.temperature).toBe(0.2);
  });

  it('deploys the rest of a trial when some params tune a study-only model', async () => {
    const user = userEvent.setup();
    const captured = mockHelix(`
optimizer:
  search_space:
    temperature: {type: fabric, path: models.default.temperature, values: [0.0, 0.2]}
    judge_temperature: {type: fabric, path: models.judge.temperature, values: [0.0, 0.5]}
models:
  judge: {provider: nvidia, model: judge-model}
`);
    renderModal({
      ...TRIAL,
      params: [...TRIAL.params, { name: 'judge_temperature', value: '0.5' }],
    });

    const dialog = await screen.findByRole('dialog');
    expect(await within(dialog).findByText('Not applied')).toBeInTheDocument();
    const deploy = within(dialog).getByRole('button', { name: 'Deploy' });
    await waitFor(() => expect(deploy).toBeEnabled());
    await user.click(deploy);

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    expect(captured.body.config?.models?.default?.temperature).toBe(0.2);
    expect(captured.body.config?.models).not.toHaveProperty('judge');
  });

  it('deploys a trial whose optimize config restates the agent model under another id', async () => {
    const user = userEvent.setup();
    const captured = mockHelix(`${OPTIMIZE_YAML}
models:
  default: {provider: nvidia, model: nvidia/llama}
`);
    renderModal();

    const dialog = await screen.findByRole('dialog');
    expect(
      await within(dialog).findByText(/The study ran the "default" model as/)
    ).toBeInTheDocument();
    const deploy = within(dialog).getByRole('button', { name: 'Deploy' });
    await waitFor(() => expect(deploy).toBeEnabled());
    await user.click(deploy);

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    expect(captured.body.config?.models?.default?.temperature).toBe(0.2);
  });

  it('blocks deploying when the study has no optimize config fileset', async () => {
    mockHelix();
    renderRoute(undefined, {
      history: getAgentOptimizationDetailRoute(workspace, 'study'),
      routes: [
        {
          path: ROUTES.workspace.agentOptimizationDetail,
          element: (
            <DeployTrialModal
              workspace={workspace}
              spec={{ strategy: 'legacy', agent: 'hermes' }}
              trial={TRIAL}
              onClose={vi.fn()}
            />
          ),
        },
      ],
    });

    const dialog = await screen.findByRole('dialog');
    expect(await within(dialog).findByText(/has no optimize config fileset/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Deploy' })).toBeDisabled();
  });
});
