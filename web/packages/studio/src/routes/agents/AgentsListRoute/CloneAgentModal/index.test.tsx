// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getAgentsListAgentsQueryKey } from '@nemo/sdk/generated/agents/agents';
import { getModelsListModelsQueryKey } from '@nemo/sdk/generated/platform/models';
import type { AgentTableRow } from '@studio/components/dataViews/AgentsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { CloneAgentModal } from '@studio/routes/agents/AgentsListRoute/CloneAgentModal';
import { getAgentsListRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const workspace = workspace1.workspace;
const MODELS_URL = `${PLATFORM_BASE_URL}${getModelsListModelsQueryKey(':workspace')[0]}`;
const CREATE_AGENT_URL = `${PLATFORM_BASE_URL}${getAgentsListAgentsQueryKey(':workspace')[0]}`;

const SOURCE_AGENT: AgentTableRow = {
  id: 'react-agent',
  name: 'react-agent',
  workspace,
  description: 'A demo agent',
  config: {
    llms: {
      llm: { _type: 'openai', model_name: 'old-model', api_key: 'x', temperature: 0 },
      embedding: { _type: 'openai', model_name: 'embed-model' },
    },
    workflow: { _type: 'react_agent', llm_name: 'llm', tool_names: ['calculator'] },
  },
  config_format: 'nat',
  created_at: '2026-04-01T00:00:00Z',
  models: ['old-model'],
  deploymentsStatus: 'No Deployments',
  deploymentsDeploying: false,
};

const mockModels = (names: string[]) => {
  server.use(
    http.get(MODELS_URL, () =>
      HttpResponse.json({
        data: names.map((name) => ({ name, workspace })),
        pagination: {
          page: 1,
          page_size: 50,
          current_page_size: names.length,
          total_pages: 1,
          total_results: names.length,
        },
        sort: '-created_at',
        filter: null,
      })
    )
  );
};

const renderModal = (sourceAgent: AgentTableRow | null = SOURCE_AGENT) =>
  renderRoute(undefined, {
    history: getAgentsListRoute(workspace),
    routes: [
      {
        path: ROUTES.workspace.agentsList,
        element: (
          <CloneAgentModal open onClose={vi.fn()} workspace={workspace} sourceAgent={sourceAgent} />
        ),
      },
      { path: ROUTES.workspace.agentDetail, element: <div>Agent detail page</div> },
    ],
  });

const getDialog = async (): Promise<HTMLElement> => {
  const dialog = await screen.findByRole('dialog');
  await within(dialog).findByRole('combobox');
  return dialog;
};

type CapturedAgent = {
  name?: string;
  description?: string;
  config_format?: string;
  config?: { llms?: Record<string, { model_name?: string }> };
};

const captureCreate = (): { body: CapturedAgent } => {
  const captured: { body: CapturedAgent } = { body: {} };
  server.use(
    http.post(CREATE_AGENT_URL, async ({ request, params }) => {
      captured.body = (await request.json()) as CapturedAgent;
      return HttpResponse.json({ ...captured.body, workspace: params['workspace'] });
    })
  );
  return captured;
};

describe('CloneAgentModal', () => {
  it("preselects the source agent's current model", async () => {
    mockModels(['new-model', 'old-model']);
    renderModal();

    const dialog = await getDialog();
    await waitFor(() =>
      expect(within(dialog).getByRole('combobox')).toHaveTextContent('old-model')
    );
  });

  it('clones with a generated name when the name field is left blank', async () => {
    const user = userEvent.setup();
    mockModels(['old-model']);
    const captured = captureCreate();

    renderModal();
    const dialog = await getDialog();
    await waitFor(() =>
      expect(within(dialog).getByRole('combobox')).toHaveTextContent('old-model')
    );
    await user.click(within(dialog).getByRole('button', { name: 'Clone' }));

    expect(await screen.findByText('Agent detail page')).toBeInTheDocument();
    expect(captured.body.name).toMatch(/^react-agent-[a-z0-9]{6}$/);
    expect(captured.body.description).toBe('A demo agent');
    expect(captured.body.config_format).toBe('nat');
    // Only the workflow's primary llm is retargeted; the embedding llm is untouched.
    expect(captured.body.config?.llms?.llm.model_name).toBe('old-model');
    expect(captured.body.config?.llms?.embedding.model_name).toBe('embed-model');
  });

  it('uses the entered name when one is provided', async () => {
    const user = userEvent.setup();
    mockModels(['old-model']);
    const captured = captureCreate();

    renderModal();
    const dialog = await getDialog();
    await waitFor(() =>
      expect(within(dialog).getByRole('combobox')).toHaveTextContent('old-model')
    );
    await user.type(within(dialog).getByRole('textbox', { name: 'Name' }), 'my-clone');
    await user.click(within(dialog).getByRole('button', { name: 'Clone' }));

    await waitFor(() => expect(captured.body.name).toBe('my-clone'));
  });

  it('clones with a different model when one is selected', async () => {
    const user = userEvent.setup();
    mockModels(['old-model', 'new-model']);
    const captured = captureCreate();

    renderModal();
    const dialog = await getDialog();
    await waitFor(() =>
      expect(within(dialog).getByRole('combobox')).toHaveTextContent('old-model')
    );

    await user.click(within(dialog).getByRole('combobox'));
    await user.click(await screen.findByRole('option', { name: 'new-model' }));
    await user.click(within(dialog).getByRole('button', { name: 'Clone' }));

    await waitFor(() => expect(captured.body.config?.llms?.llm.model_name).toBe('new-model'));
  });

  it('keeps a name typed before the models query resolves', async () => {
    const user = userEvent.setup();
    // Gate the models response so seeding fires only after the user has typed a name.
    let releaseModels!: () => void;
    const modelsLoaded = new Promise<void>((resolve) => {
      releaseModels = resolve;
    });
    server.use(
      http.get(MODELS_URL, async () => {
        await modelsLoaded;
        return HttpResponse.json({
          data: [{ name: 'old-model', workspace }],
          pagination: {
            page: 1,
            page_size: 50,
            current_page_size: 1,
            total_pages: 1,
            total_results: 1,
          },
          sort: '-created_at',
          filter: null,
        });
      })
    );

    renderModal();
    const dialog = await screen.findByRole('dialog');
    const nameInput = await within(dialog).findByRole('textbox', { name: 'Name' });
    await user.type(nameInput, 'my-typed-name');
    expect(nameInput).toHaveValue('my-typed-name');

    // Models resolve now → the effect seeds the model. The typed name must survive.
    releaseModels();
    await waitFor(() =>
      expect(within(dialog).getByRole('combobox')).toHaveTextContent('old-model')
    );
    expect(nameInput).toHaveValue('my-typed-name');
  });
});
