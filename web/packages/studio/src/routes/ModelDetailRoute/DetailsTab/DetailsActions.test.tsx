// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelsDeleteModel } from '@nemo/sdk/generated/platform/models';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { ROUTES } from '@studio/constants/routes';
import { DetailsActions } from '@studio/routes/ModelDetailRoute/DetailsTab/DetailsActions';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import type { ReactElement } from 'react';

const mockEnvironment = vi.hoisted(() => ({ customizerEnabled: true }));
const mockNavigate = vi.hoisted(() => vi.fn());

vi.mock('@studio/constants/environment', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/constants/environment')>()),
  get CUSTOMIZER_ENABLED() {
    return mockEnvironment.customizerEnabled;
  },
}));
vi.mock('react-router', async (importOriginal) => ({
  ...(await importOriginal<typeof import('react-router')>()),
  useNavigate: () => mockNavigate,
}));
vi.mock('@nemo/sdk/generated/platform/models');

const model = { name: 'my-model', workspace: 'default', fileset: 'ws/weights' } as ModelEntity;

const render = (element: ReactElement) =>
  renderRoute(undefined, {
    history: '/workspaces/ws/base-models/my-model',
    routes: [{ path: ROUTES.workspace.baseModelsModel, element }],
  });

describe('DetailsActions', () => {
  const deleteModel = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    mockEnvironment.customizerEnabled = true;
    vi.mocked(useModelsDeleteModel).mockReturnValue({
      mutateAsync: deleteModel,
    } as unknown as ReturnType<typeof useModelsDeleteModel>);
  });

  it('links to the fine-tuning form in the workspace being viewed', async () => {
    render(<DetailsActions model={model} canDelete={false} />);

    expect(await screen.findByRole('link', { name: 'Customize' })).toHaveAttribute(
      'href',
      expect.stringContaining('/workspaces/ws/fine-tune/new')
    );
  });

  it('disables customize for a model without weights', async () => {
    render(<DetailsActions model={{ ...model, fileset: undefined }} canDelete={false} />);

    expect(await screen.findByRole('button', { name: 'Customize' })).toBeDisabled();
  });

  it('hides customize when fine-tuning is off', async () => {
    mockEnvironment.customizerEnabled = false;
    render(<DetailsActions model={model} canDelete />);

    expect(await screen.findByRole('button', { name: 'Delete' })).toBeInTheDocument();
    expect(screen.queryByText('Customize')).not.toBeInTheDocument();
  });

  it('hides delete unless allowed', async () => {
    render(<DetailsActions model={model} canDelete={false} />);

    await screen.findByRole('link', { name: 'Customize' });
    expect(screen.queryByRole('button', { name: 'Delete' })).not.toBeInTheDocument();
  });

  it('deletes the model and returns to the catalog it was opened from', async () => {
    render(<DetailsActions model={model} canDelete />);

    await userEvent.click(await screen.findByRole('button', { name: 'Delete' }));
    const dialog = await screen.findByRole('dialog');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Delete' }));

    expect(deleteModel).toHaveBeenCalledWith({ workspace: 'default', name: 'my-model' });
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith('/workspaces/ws/base-models', { replace: true })
    );
  });
});
