// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getAgentsListAgentsQueryKey } from '@nemo/sdk/generated/agents/agents';
import {
  getFilesListFilesetsQueryKey,
  getFilesRetrieveFilesetQueryKey,
} from '@nemo/sdk/generated/platform/files';
import { getSecretsListSecretsQueryKey } from '@nemo/sdk/generated/platform/secrets';
import { CreateFilesetStart } from '@studio/components/CreateFilesetStart';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { render, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('@studio/constants/environment', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/constants/environment')>()),
  PERSONA_TEST_SETS_ENABLED: true,
}));

const workspace = workspace1.workspace;
const agentName = 'calculator-agent';
const PERSONAS_FILESET = 'nemotron-personas-dataset-en_us';

const mockApis = (personasExist: boolean) => {
  let registered = personasExist;
  server.use(
    http.get(mockApiUrl(getAgentsListAgentsQueryKey, ':workspace'), ({ request }) => {
      const page = Number(new URL(request.url).searchParams.get('page') ?? 1);
      const agent = { name: page === 1 ? 'first-page-agent' : agentName, workspace };
      return HttpResponse.json({
        data: [
          page === 1
            ? agent
            : {
                ...agent,
                config: { instructions: { system: { content: 'Return only the answer.' } } },
              },
        ],
        pagination: { total: 2, total_pages: 2, page, page_size: 1 },
      });
    }),
    http.get(mockApiUrl(getFilesRetrieveFilesetQueryKey, 'system', ':name'), () =>
      registered
        ? HttpResponse.json({ name: PERSONAS_FILESET, workspace: 'system' })
        : HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    ),
    http.post(mockApiUrl(getFilesListFilesetsQueryKey, 'system'), () => {
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
};

const openPanel = async () => {
  const user = userEvent.setup();
  const onContinue = vi.fn();
  render(<CreateFilesetStart workspace={workspace} onContinue={onContinue} />);
  await user.click(screen.getByRole('radio', { name: /Personas/ }));
  return { user, onContinue };
};

const pickAgent = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(await screen.findByRole('combobox', { name: /Agent/ }));
  await user.click(await screen.findByRole('option', { name: agentName }));
};

describe('PersonaTestSetPanel', () => {
  it('continues with a persona job drafted for the selected agent', async () => {
    mockApis(true);
    const { user, onContinue } = await openPanel();

    const next = screen.getByRole('button', { name: /continue/i });
    expect(next).toBeDisabled();
    await pickAgent(user);
    await waitFor(() => expect(next).toBeEnabled());
    await user.click(next);

    const [selection] = onContinue.mock.calls[0];
    expect(selection.optionId).toBe('persona');
    expect(selection.jobRequest.spec.config.columns[0]).toMatchObject({
      sampler_type: 'person',
      params: { locale: 'en_US', with_synthetic_personas: true },
    });
    expect(selection.jobRequest.custom_fields).toEqual({ persona_test_set_agent: agentName });
  });

  it('blocks continuing until the persona dataset is registered', async () => {
    mockApis(false);
    const { user } = await openPanel();

    await pickAgent(user);
    await screen.findByText("Nemotron Personas for en_US isn't set up");
    expect(screen.getByRole('button', { name: /continue/i })).toBeDisabled();

    await user.click(screen.getByRole('combobox', { name: /NGC API key secret/ }));
    await user.click(await screen.findByRole('option', { name: 'ngc-api-key' }));
    await user.click(screen.getByRole('button', { name: 'Register dataset' }));

    await waitFor(() => expect(screen.getByRole('button', { name: /continue/i })).toBeEnabled());
  });

  it('disables the Personas tile, with a tooltip, when the workspace has no agents', async () => {
    mockApis(true);
    server.use(
      http.get(mockApiUrl(getAgentsListAgentsQueryKey, ':workspace'), () =>
        HttpResponse.json({
          data: [],
          pagination: { total: 0, total_pages: 0, page: 1, page_size: 1 },
        })
      )
    );
    const user = userEvent.setup();
    render(<CreateFilesetStart workspace={workspace} onContinue={vi.fn()} />);

    // Wrapping it for the tooltip remounts the tile, so query it fresh.
    await waitFor(() => expect(screen.getByRole('radio', { name: /Personas/ })).toBeDisabled());
    await user.hover(screen.getByText('Personas'));
    expect((await screen.findAllByText(/Create an agent first/)).length).toBeGreaterThan(0);
  });

  it('shows agent-list failures and recovers on retry', async () => {
    mockApis(true);
    let fail = true;
    server.use(
      http.get(mockApiUrl(getAgentsListAgentsQueryKey, ':workspace'), () =>
        fail
          ? HttpResponse.json({ detail: 'boom' }, { status: 500 })
          : HttpResponse.json({
              data: [{ name: agentName, workspace }],
              pagination: { total: 1, total_pages: 1, page: 1, page_size: 1 },
            })
      )
    );
    const { user } = await openPanel();

    await screen.findByText('Could not load agents.');
    fail = false;
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    await pickAgent(user);
  });
});
