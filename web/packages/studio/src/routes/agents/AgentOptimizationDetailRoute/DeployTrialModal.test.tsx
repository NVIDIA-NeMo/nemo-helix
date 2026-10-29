// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RunStrategyJob } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategyJob';
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
const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;
const FILE_URL = `${FILESETS_URL}/:name/-/:path`;
const DEPLOYMENTS_URL = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments`;

const SPEC = {
  strategy: 'config-tuning',
  agent: 'hermes',
  optimize_config: 'optimize.yaml',
  optimize_config_fileset: 'hermes-bundle',
};

const JOB: RunStrategyJob = { name: 'study', spec: SPEC, created_at: '2026-04-02T00:00:00Z' };

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
  telemetry: { enabled: true, agent_name: 'hermes' },
};

const TRIAL: Trial = {
  number: 4,
  phase: 'numeric',
  state: 'COMPLETE',
  durationSeconds: 12,
  paretoOptimal: true,
  metrics: [],
  params: [{ name: 'temperature', value: '0.2' }],
};

interface CapturedAgent {
  name?: string;
  config_format?: string;
  config?: {
    name?: string;
    models?: Record<string, { model?: string; temperature?: number }>;
    telemetry?: { agent_name?: string };
  };
}

interface Captured {
  agent: CapturedAgent;
  bundleWorkspace?: string;
  createdFilesets: string[];
}

const mockHelix = ({
  optimizeYaml = OPTIMIZE_YAML,
  agentUpdatedAt = '2026-04-01T00:00:00Z',
}: { optimizeYaml?: string; agentUpdatedAt?: string } = {}): Captured => {
  const captured: Captured = { agent: {}, createdFilesets: [] };
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
        updated_at: agentUpdatedAt,
      })
    ),
    // Neither the source agent's spec fileset nor the new agent's exists.
    http.get(`${FILESETS_URL}/:name`, () => new HttpResponse(null, { status: 404 })),
    http.post(FILESETS_URL, async ({ request }) => {
      captured.createdFilesets.push(((await request.json()) as { name: string }).name);
      return HttpResponse.json({});
    }),
    http.get(FILE_URL, ({ params }) => {
      captured.bundleWorkspace = String(params['workspace']);
      return new HttpResponse(optimizeYaml);
    }),
    http.post(AGENTS_URL, async ({ request }) => {
      captured.agent = (await request.json()) as CapturedAgent;
      return HttpResponse.json({ ...captured.agent, workspace });
    })
  );
  return captured;
};

const renderModal = (trial: Trial = TRIAL, job: RunStrategyJob = JOB) =>
  renderRoute(undefined, {
    history: getAgentOptimizationDetailRoute(workspace, 'study'),
    routes: [
      {
        path: ROUTES.workspace.agentOptimizationDetail,
        element: (
          <DeployTrialModal workspace={workspace} job={job} trial={trial} onClose={vi.fn()} />
        ),
      },
      { path: ROUTES.workspace.agentDetail, element: <div>Agent detail page</div> },
    ],
  });

const deployButton = async () => {
  const dialog = await screen.findByRole('dialog');
  // The loading spinner is part of the button's name until the agent and config load.
  const deploy = await within(dialog).findByRole('button', { name: 'Deploy' });
  await waitFor(() => expect(deploy).toBeEnabled());
  return { dialog, deploy };
};

describe('DeployTrialModal', () => {
  it('creates an agent with the trial configuration and opens it', async () => {
    const user = userEvent.setup();
    const captured = mockHelix();
    renderModal();

    const { dialog, deploy } = await deployButton();
    expect(within(dialog).getByRole('textbox')).toHaveValue('hermes-trial-4');
    expect(within(dialog).getByText('models.default.temperature')).toBeInTheDocument();
    await user.click(deploy);

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    expect(captured.bundleWorkspace).toBe(workspace);
    expect(captured.createdFilesets).toEqual([]);
    expect(captured.agent.name).toBe('hermes-trial-4');
    expect(captured.agent.config_format).toBe('nemo-agents-spec-v1');
    expect(captured.agent.config?.name).toBe('hermes-trial-4');
    expect(captured.agent.config?.telemetry?.agent_name).toBe('hermes-trial-4');
    expect(captured.agent.config?.models?.default?.temperature).toBe(0.2);
  });

  it("shows an error when the name doesn't start with a lowercase letter", async () => {
    const user = userEvent.setup();
    mockHelix();
    renderModal();

    const { dialog } = await deployButton();
    const name = within(dialog).getByRole('textbox');
    await user.clear(name);
    await user.type(name, '---');
    await user.tab();

    expect(
      await within(dialog).findByText('Name must start with a lowercase letter.')
    ).toBeInTheDocument();
  });

  it('adds the models only the optimize config defines', async () => {
    const user = userEvent.setup();
    const captured = mockHelix({
      optimizeYaml: `${OPTIMIZE_YAML}
models:
  fast: {provider: nvidia, model: llama-mini}
`,
    });
    renderModal();

    const { dialog, deploy } = await deployButton();
    expect(within(dialog).getByText(/Also adds "fast" model/)).toBeInTheDocument();
    await user.click(deploy);

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    expect(captured.agent.config?.models?.fast).toEqual({
      provider: 'nvidia',
      model: 'llama-mini',
    });
  });

  it('leaves out the judge model the study only scored with', async () => {
    const user = userEvent.setup();
    const captured = mockHelix({
      optimizeYaml: `${OPTIMIZE_YAML}
models:
  judge: {provider: nvidia, model: judge-model}
eval:
  evaluators:
    accuracy: {_type: tunable_rag_evaluator, llm_name: judge}
`,
    });
    renderModal();

    const { dialog, deploy } = await deployButton();
    expect(within(dialog).queryByText(/Also adds/)).not.toBeInTheDocument();
    await user.click(deploy);

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    expect(captured.agent.config?.models).not.toHaveProperty('judge');
  });

  it('warns when the study ran the agent model with different settings', async () => {
    const user = userEvent.setup();
    const captured = mockHelix({
      optimizeYaml: `${OPTIMIZE_YAML}
models:
  default: {provider: nvidia, model: nvidia/llama, temperature: 0.7}
`,
    });
    renderModal();

    const { dialog, deploy } = await deployButton();
    expect(
      within(dialog).getByText(/The study ran the "default" model with different settings/)
    ).toHaveTextContent('(model)');
    await user.click(deploy);

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    expect(captured.agent.config?.models?.default).toMatchObject({
      model: 'llama',
      temperature: 0.2,
    });
  });

  it('warns when the agent changed after the study ran', async () => {
    mockHelix({ agentUpdatedAt: '2026-04-03T00:00:00Z' });
    renderModal();

    const { dialog } = await deployButton();
    expect(within(dialog).getByText(/was changed after this study ran/)).toBeInTheDocument();
  });

  it('blocks deploying when the study has no optimize config fileset', async () => {
    mockHelix();
    renderModal(TRIAL, { ...JOB, spec: { strategy: 'config-tuning', agent: 'hermes' } });

    const dialog = await screen.findByRole('dialog');
    expect(await within(dialog).findByText(/has no optimize config fileset/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Deploy' })).toBeDisabled();
  });
});

describe('DeployTrialModal deploy after create', () => {
  const captureDeployments = () => {
    const deployments: { agent?: string; deployment_mode?: string }[] = [];
    server.use(
      http.post(DEPLOYMENTS_URL, async ({ request }) => {
        const body = (await request.json()) as { agent?: string; deployment_mode?: string };
        deployments.push(body);
        return HttpResponse.json({ name: 'deployment', ...body });
      })
    );
    return deployments;
  };

  it('deploys the new agent as a subprocess by default', async () => {
    const user = userEvent.setup();
    mockHelix();
    const deployments = captureDeployments();
    renderModal();

    const { dialog, deploy } = await deployButton();
    expect(
      await within(dialog).findByRole('checkbox', { name: 'Deploy after creating' })
    ).toBeChecked();
    await user.click(deploy);

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    await waitFor(() =>
      expect(deployments).toEqual([{ agent: 'hermes-trial-4', deployment_mode: 'subprocess' }])
    );
  });

  it('creates without deploying when the box is cleared', async () => {
    const user = userEvent.setup();
    const captured = mockHelix();
    const deployments = captureDeployments();
    renderModal();

    const { dialog, deploy } = await deployButton();
    await user.click(
      await within(dialog).findByRole('checkbox', { name: 'Deploy after creating' })
    );
    await user.click(deploy);

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    expect(captured.agent.name).toBe('hermes-trial-4');
    expect(deployments).toEqual([]);
  });
});
