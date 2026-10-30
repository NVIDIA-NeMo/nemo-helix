// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getFilesListFilesetsQueryKey,
  getFilesRetrieveFilesetQueryKey,
} from '@nemo/sdk/generated/platform/files';
import { getSecretsListSecretsQueryKey } from '@nemo/sdk/generated/platform/secrets';
import { PersonaTestSetModal } from '@studio/components/agents/PersonaTestSet/PersonaTestSetModal';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useLocation } from 'react-router';

const workspace = workspace1.workspace;
const agentName = 'calculator-agent';
const PERSONAS_FILESET = 'nemotron-personas-dataset-en_us';

const mockPersonasFileset = (exists: boolean) => {
  const created: unknown[] = [];
  let registered = exists;
  server.use(
    http.get(mockApiUrl(getFilesRetrieveFilesetQueryKey, 'system', ':name'), ({ params }) =>
      registered && params.name === PERSONAS_FILESET
        ? HttpResponse.json({ name: PERSONAS_FILESET, workspace: 'system' })
        : HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    ),
    http.post(mockApiUrl(getFilesListFilesetsQueryKey, 'system'), async ({ request }) => {
      created.push(await request.json());
      registered = true;
      return HttpResponse.json({ name: PERSONAS_FILESET, workspace: 'system' });
    }),
    http.get(mockApiUrl(getSecretsListSecretsQueryKey, ':workspace'), () =>
      HttpResponse.json({
        data: [{ name: 'ngc-api-key', workspace }],
        pagination: { total: 1, page: 1, page_size: 1000 },
      })
    )
  );
  return created;
};

const BuilderState = () => {
  const { pathname, state } = useLocation();
  return <pre data-testid="builder-state">{JSON.stringify({ pathname, state })}</pre>;
};

const renderModal = () =>
  renderRoute(undefined, {
    history: getAgentDetailRoute(workspace, agentName),
    routes: [
      {
        path: ROUTES.workspace.agentDetail,
        element: (
          <PersonaTestSetModal
            open
            onClose={vi.fn()}
            workspace={workspace}
            agent={{ name: agentName, instructions: 'Return only the answer.' }}
          />
        ),
      },
      { path: ROUTES.workspace.dataDesignerJobBuild, element: <BuilderState /> },
    ],
  });

describe('PersonaTestSetModal', () => {
  it('opens the drafted persona job in the Data Designer builder', async () => {
    mockPersonasFileset(true);
    const user = userEvent.setup();
    renderModal();

    const submit = await screen.findByRole('button', { name: 'Review in Data Designer' });
    await waitFor(() => expect(submit).toBeEnabled());
    await user.click(submit);

    const { pathname, state } = JSON.parse(
      (await screen.findByTestId('builder-state')).textContent ?? '{}'
    );
    expect(pathname).toBe(`/workspaces/${workspace}/data-designer/new/build`);
    expect(state.generatedJobRequest.spec.config.columns[0]).toMatchObject({
      sampler_type: 'person',
      params: { locale: 'en_US', with_synthetic_personas: true },
    });
    expect(state.generatedJobRequest.custom_fields).toEqual({
      persona_test_set_agent: agentName,
    });
  });

  it('registers a missing persona dataset before it lets the job start', async () => {
    const created = mockPersonasFileset(false);
    const user = userEvent.setup();
    renderModal();

    const banner = await screen.findByText("Nemotron Personas for en_US isn't set up");
    const submit = screen.getByRole('button', { name: 'Review in Data Designer' });
    expect(submit).toBeDisabled();

    const dialog = screen.getByRole('dialog', { name: 'Generate persona test set' });
    await user.click(within(dialog).getByRole('combobox', { name: /NGC API key secret/ }));
    await user.click(await screen.findByRole('option', { name: 'ngc-api-key' }));
    await user.click(within(dialog).getByRole('button', { name: 'Register dataset' }));

    await waitFor(() => expect(submit).toBeEnabled());
    expect(banner).not.toBeInTheDocument();
    expect(created).toEqual([
      expect.objectContaining({
        name: PERSONAS_FILESET,
        storage: expect.objectContaining({
          type: 'ngc',
          api_key_secret: `${workspace}/ngc-api-key`,
          team: 'nemotron-personas',
          target: PERSONAS_FILESET,
        }),
      }),
    ]);
  });
});
