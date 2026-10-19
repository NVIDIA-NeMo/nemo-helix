// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { GuardrailConfig } from '@nemo/sdk/generated/platform/schema';
import { DEFAULT_BUILD_MODEL_NAME } from '@studio/constants/constants';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { CreateGuardrailModal } from '@studio/routes/guardrails/CreateGuardrailModal';
import { mockUseNavigate, mockUseParams } from '@studio/tests/util/mockUseParams';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { BrowserRouter } from 'react-router';
import type { Mock } from 'vitest';

const WORKSPACE = 'test-workspace';
const CONFIGS_URL = `${PLATFORM_BASE_URL}/apis/guardrails/v2/workspaces/:workspace/configs`;

// The modal seeds a main model from the workspace catalogue. Stubbed rather than served over
// MSW so each test states the catalogue it wants; `modelGroups` is what the hook returns.
let modelGroups: unknown[] = [];
const useModelsFromWorkspaceSpy = vi.fn();
vi.mock('@nemo/common/src/api/models/useModelsFromWorkspace', () => ({
  useModelsFromWorkspace: (options: unknown) => {
    useModelsFromWorkspaceSpy(options);
    return { groups: modelGroups };
  },
}));

const MODEL_GROUPS = [
  {
    workspace: WORKSPACE,
    models: [
      { name: 'adept-fuyu-8b', workspace: WORKSPACE, model_providers: ['p'] },
      { name: DEFAULT_BUILD_MODEL_NAME, workspace: WORKSPACE, model_providers: ['p'] },
    ],
  },
];

const SOURCE_CONFIG: GuardrailConfig = {
  name: 'my-rail',
  workspace: WORKSPACE,
  description: 'Blocks unsafe content',
  data: {
    models: [
      { engine: 'nim', model: 'nvidia/llama-3.1-nemoguard-8b-content-safety', type: 'main' },
    ],
  },
  id: 'guardrail-config-1',
  entity_id: 'guardrail-config-1',
  parent: `workspace-${WORKSPACE}`,
  created_at: '2026-01-01T00:00:00Z',
  created_by: null,
  updated_at: '2026-01-01T00:00:00Z',
  updated_by: null,
  db_version: 1,
};

const renderModal = ({
  onClose = vi.fn(),
  sourceConfig,
}: { onClose?: Mock; sourceConfig?: GuardrailConfig } = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <CreateGuardrailModal open onClose={onClose} sourceConfig={sourceConfig} />
      </BrowserRouter>
    </QueryClientProvider>
  );
  return { onClose };
};

describe('CreateGuardrailModal', () => {
  let navigate: Mock;

  beforeEach(() => {
    mockUseParams({ [ROUTE_PARAMS.workspace]: WORKSPACE });
    navigate = vi.fn();
    mockUseNavigate(navigate);
    modelGroups = [];
    useModelsFromWorkspaceSpy.mockClear();
  });

  it('seeds a fresh config with the resolved main model', async () => {
    modelGroups = MODEL_GROUPS;
    let body: unknown;
    server.use(
      http.post(CONFIGS_URL, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json({ name: 'my-rail' }, { status: 201 });
      })
    );
    const user = userEvent.setup();
    renderModal();

    await user.type(screen.getByRole('textbox'), 'my-rail');
    await user.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(body).toEqual({
        name: 'my-rail',
        data: {
          models: [
            {
              type: 'main',
              engine: 'nim',
              mode: 'chat',
              model: `${WORKSPACE}/${DEFAULT_BUILD_MODEL_NAME}`,
            },
          ],
        },
      });
    });
  });

  it('creates without models when the workspace serves none', async () => {
    let body: unknown;
    server.use(
      http.post(CONFIGS_URL, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json({ name: 'my-rail' }, { status: 201 });
      })
    );
    const user = userEvent.setup();
    renderModal();

    await user.type(screen.getByRole('textbox'), 'my-rail');
    await user.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(body).toEqual({ name: 'my-rail' });
    });
  });

  it('creates the guardrail and navigates to its detail page', async () => {
    server.use(
      http.post(CONFIGS_URL, () => HttpResponse.json({ name: 'my-rail' }, { status: 201 }))
    );
    const user = userEvent.setup();
    const { onClose } = renderModal();

    await user.type(screen.getByRole('textbox'), 'my-rail');
    await user.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith(`/workspaces/${WORKSPACE}/guardrails/my-rail`);
    });
    expect(onClose).toHaveBeenCalled();
  });

  it('keeps the modal open and does not navigate when creation fails', async () => {
    server.use(
      http.post(CONFIGS_URL, () =>
        HttpResponse.json({ detail: 'Name already in use' }, { status: 409 })
      )
    );
    const user = userEvent.setup();
    const { onClose } = renderModal();

    await user.type(screen.getByRole('textbox'), 'my-rail');
    await user.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(screen.getByText('Name already in use')).toBeInTheDocument();
    });
    expect(navigate).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('rejects a name that does not match the entity name pattern', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.type(screen.getByRole('textbox'), 'Not Valid');

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();
    });
  });

  describe('with a source config', () => {
    it('defaults the name to <source>-copy', async () => {
      renderModal({ sourceConfig: SOURCE_CONFIG });

      await waitFor(() => {
        expect(screen.getByRole('textbox')).toHaveValue('my-rail-copy');
      });
      expect(screen.getByText('Duplicate Guardrail')).toBeInTheDocument();
    });

    it('does not query the model catalogue', async () => {
      renderModal({ sourceConfig: SOURCE_CONFIG });

      // Settle the modal's open effect (name reset + focus) before asserting.
      await waitFor(() => {
        expect(screen.getByRole('textbox')).toHaveValue('my-rail-copy');
      });
      // The duplicate carries the source's models; resolving a default would be wasted work.
      expect(useModelsFromWorkspaceSpy).toHaveBeenCalledWith(
        expect.objectContaining({ queryOptions: { enabled: false } })
      );
    });

    it('creates a copy carrying the source description and data', async () => {
      modelGroups = MODEL_GROUPS;
      let body: unknown;
      server.use(
        http.post(CONFIGS_URL, async ({ request }) => {
          body = await request.json();
          return HttpResponse.json({ name: 'my-rail-copy' }, { status: 201 });
        })
      );
      const user = userEvent.setup();
      const { onClose } = renderModal({ sourceConfig: SOURCE_CONFIG });

      await user.click(screen.getByRole('button', { name: 'Create' }));

      await waitFor(() => {
        expect(navigate).toHaveBeenCalledWith(`/workspaces/${WORKSPACE}/guardrails/my-rail-copy`);
      });
      expect(body).toEqual({
        name: 'my-rail-copy',
        description: SOURCE_CONFIG.description,
        data: SOURCE_CONFIG.data,
      });
      expect(onClose).toHaveBeenCalled();
    });
  });
});
