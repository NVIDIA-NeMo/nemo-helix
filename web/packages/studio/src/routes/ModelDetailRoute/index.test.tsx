// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { ROUTES } from '@studio/constants/routes';
import { useCustomizationJob } from '@studio/hooks/useCustomizationJob';
import { type ModelLineage, useModelLineage } from '@studio/hooks/useModelLineage';
import { ModelDetailRoute } from '@studio/routes/ModelDetailRoute';
import { renderRoute } from '@studio/tests/util/render';
import { screen } from '@testing-library/react';

vi.mock('@studio/hooks/useModelLineage', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/hooks/useModelLineage')>()),
  useModelLineage: vi.fn(),
}));
vi.mock('@studio/hooks/useCustomizationJob');
vi.mock('@studio/routes/ModelDetailRoute/DetailsTab', () => ({
  DetailsTab: () => <div>details content</div>,
}));
vi.mock('@studio/routes/ModelDetailRoute/FilesTab', () => ({
  FilesTab: () => <div>files content</div>,
}));
vi.mock('@studio/components/ModelChatTab', () => ({ ModelChatTab: () => <div /> }));
vi.mock('@studio/components/CustomizationOverview', () => ({
  CustomizationOverview: () => <div>training content</div>,
}));
vi.mock('@studio/routes/CustomizationJobDetailsRoute/LogsTab', () => ({ LogsTab: () => <div /> }));

const adapter = { name: 'my-adapter', workspace: 'ws', finetuning_type: 'lora' } as Adapter;
const model = {
  name: 'my-model',
  workspace: 'ws',
  base_model: 'ws/base',
  finetuning_type: 'lora',
  adapters: [adapter],
} as ModelEntity;

const mockLineage = (overrides: Partial<ModelLineage> = {}) =>
  vi.mocked(useModelLineage).mockReturnValue({
    model,
    adapter: null,
    isLoading: false,
    isError: false,
    isAdapterMissing: false,
    job: { jobName: 'job-1', workspace: 'ws', isLoading: false },
    fileset: null,
    deployment: { kind: 'not-deployed' },
    ...overrides,
  });

const renderPage = (path: string) =>
  renderRoute(undefined, {
    history: path,
    routes: [
      { path: ROUTES.workspace.modelDetail, element: <ModelDetailRoute /> },
      { path: ROUTES.workspace.modelAdapterDetail, element: <ModelDetailRoute /> },
    ],
  });

describe('ModelDetailRoute', () => {
  beforeEach(() => {
    vi.mocked(useCustomizationJob).mockReturnValue({
      job: undefined,
    } as unknown as ReturnType<typeof useCustomizationJob>);
  });

  it('shows the model with its tabs, opening on Details', async () => {
    mockLineage();
    renderPage('/workspaces/ws/fine-tune/models/my-model');

    expect(await screen.findByTestId('nv-page-header-heading')).toHaveTextContent('my-model');
    for (const name of ['Details', 'Chat', 'Files', 'Training', 'Logs']) {
      expect(screen.getByRole('tab', { name })).toBeInTheDocument();
    }
    expect(screen.getAllByRole('tab')).toHaveLength(5);
    expect(screen.getByText('details content')).toBeInTheDocument();
  });

  it('opens the tab named in the URL, with its actions', async () => {
    mockLineage();
    renderPage('/workspaces/ws/fine-tune/models/my-model?tab=training');

    expect(await screen.findByText('training content')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open Job' })).toHaveAttribute(
      'href',
      expect.stringContaining('/fine-tune/job-1')
    );
  });

  it('links out to the fileset from the Files tab', async () => {
    mockLineage({ fileset: { workspace: 'files-ws', name: 'weights' } });
    renderPage('/workspaces/ws/fine-tune/models/my-model?tab=files');

    expect(await screen.findByText('files content')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open Fileset' })).toHaveAttribute(
      'href',
      expect.stringContaining('weights')
    );
  });

  it('keeps the Files tab without a fileset, with nothing to open', async () => {
    mockLineage({ fileset: null });
    renderPage('/workspaces/ws/fine-tune/models/my-model?tab=files');

    expect(await screen.findByText('files content')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Open Fileset' })).not.toBeInTheDocument();
  });

  it('disables the job tabs while the job lookup runs', async () => {
    mockLineage({ job: { jobName: undefined, workspace: 'ws', isLoading: true } });
    renderPage('/workspaces/ws/fine-tune/models/my-model');

    expect(await screen.findByRole('tab', { name: 'Training' })).toBeDisabled();
    expect(screen.getByRole('tab', { name: 'Logs' })).toBeDisabled();
  });

  it('drops the job tabs when no job is found, falling back to Details', async () => {
    mockLineage({ job: { jobName: undefined, workspace: 'ws', isLoading: false } });
    renderPage('/workspaces/ws/fine-tune/models/my-model?tab=training');

    expect(await screen.findByText('details content')).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Training' })).not.toBeInTheDocument();
  });

  it('shows an adapter under its own name', async () => {
    mockLineage({ adapter });
    renderPage('/workspaces/ws/fine-tune/models/my-model/adapters/my-adapter');

    expect(await screen.findByTestId('nv-page-header-heading')).toHaveTextContent('my-adapter');
    expect(screen.getByText('Model: my-model')).toBeInTheDocument();
  });

  it('reports a model that cannot be loaded', async () => {
    mockLineage({ model: undefined, isError: true });
    renderPage('/workspaces/ws/fine-tune/models/my-model');

    expect(await screen.findByText('Model not found')).toBeInTheDocument();
  });

  it('reports an adapter the model does not have', async () => {
    mockLineage({ isAdapterMissing: true });
    renderPage('/workspaces/ws/fine-tune/models/my-model/adapters/missing');

    expect(await screen.findByText('Adapter not found')).toBeInTheDocument();
  });
});
