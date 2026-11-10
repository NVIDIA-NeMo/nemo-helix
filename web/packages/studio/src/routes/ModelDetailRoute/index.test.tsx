// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { ROUTES } from '@studio/constants/routes';
import { useCustomizationJob } from '@studio/hooks/useCustomizationJob';
import { type ModelLineage, useModelLineage } from '@studio/hooks/useModelLineage';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { ModelDetailRoute } from '@studio/routes/ModelDetailRoute';
import { renderRoute } from '@studio/tests/util/render';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@studio/hooks/useModelLineage', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/hooks/useModelLineage')>()),
  useModelLineage: vi.fn(),
}));
vi.mock('@studio/hooks/useCustomizationJob');
vi.mock('@studio/providers/breadcrumbs/useBreadcrumbs', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/providers/breadcrumbs/useBreadcrumbs')>()),
  useBreadcrumbs: vi.fn(),
}));
vi.mock('@studio/routes/ModelDetailRoute/DetailsTab/DetailsActions', () => ({
  DetailsActions: ({ canDelete }: { canDelete: boolean }) => (
    <button type="button">details actions{canDelete ? ' with delete' : ''}</button>
  ),
}));
vi.mock('@studio/routes/ModelDetailRoute/DetailsTab', () => ({
  DetailsTab: () => <div>details content</div>,
}));
vi.mock('@studio/routes/ModelDetailRoute/FilesTab', () => ({
  FilesTab: () => <div>files content</div>,
}));
vi.mock('@studio/routes/ModelDetailRoute/DeploymentTab', () => ({
  DeploymentTab: () => <div>deployment content</div>,
}));
vi.mock('@studio/routes/ModelDetailRoute/DeploymentTab/DeploymentActions', () => ({
  DeploymentActions: () => <button type="button">deployment actions</button>,
}));
vi.mock('@studio/routes/ModelDetailRoute/ChatTab', () => ({ ChatTab: () => <div /> }));
vi.mock('@studio/components/dataViews/EvaluationResultsDataView', () => ({
  EvaluationResultsDataView: ({ targetModelName }: { targetModelName?: string }) => (
    <div>evaluations for {targetModelName}</div>
  ),
}));
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
    baseDeployment: { kind: 'not-deployed' },
    ...overrides,
  });

const renderPage = (path: string) =>
  renderRoute(undefined, {
    history: path,
    routes: [
      { path: ROUTES.workspace.modelDetail, element: <ModelDetailRoute /> },
      { path: ROUTES.workspace.modelAdapterDetail, element: <ModelDetailRoute /> },
      { path: ROUTES.workspace.baseModelsModel, element: <ModelDetailRoute parent="catalog" /> },
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
    for (const name of [
      'Details',
      'Chat',
      'Evaluations',
      'Files',
      'Deployment',
      'Training',
      'Logs',
    ]) {
      expect(screen.getByRole('tab', { name })).toBeInTheDocument();
    }
    expect(screen.getAllByRole('tab')).toHaveLength(7);
    expect(screen.getByText('details content')).toBeInTheDocument();
  });

  it('opens the tab named in the URL, with its actions', async () => {
    mockLineage();
    renderPage('/workspaces/ws/fine-tune/models/my-model/training');

    expect(await screen.findByText('training content')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open Job' })).toHaveAttribute(
      'href',
      expect.stringContaining('/fine-tune/job-1')
    );
  });

  it('links out to the fileset from the Files tab', async () => {
    mockLineage({ fileset: { workspace: 'files-ws', name: 'weights' } });
    renderPage('/workspaces/ws/fine-tune/models/my-model/files');

    expect(await screen.findByText('files content')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open Fileset' })).toHaveAttribute(
      'href',
      expect.stringContaining('weights')
    );
  });

  it('keeps the Files tab without a fileset, with nothing to open', async () => {
    mockLineage({ fileset: null });
    renderPage('/workspaces/ws/fine-tune/models/my-model/files');

    expect(await screen.findByText('files content')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Open Fileset' })).not.toBeInTheDocument();
  });

  it('moves between tabs through the URL path', async () => {
    mockLineage();
    renderPage('/workspaces/ws/fine-tune/models/my-model');

    await userEvent.click(await screen.findByRole('tab', { name: 'Deployment' }));
    expect(await screen.findByText('deployment content')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('tab', { name: 'Details' }));
    expect(await screen.findByText('details content')).toBeInTheDocument();
  });

  it('puts the deployment actions in the page header on the Deployment tab only', async () => {
    mockLineage();
    renderPage('/workspaces/ws/fine-tune/models/my-model/deployment');

    expect(await screen.findByText('deployment content')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'deployment actions' })).toBeInTheDocument();
  });

  it('leaves the deployment actions off other tabs', async () => {
    mockLineage();
    renderPage('/workspaces/ws/fine-tune/models/my-model');

    expect(await screen.findByText('details content')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'deployment actions' })).not.toBeInTheDocument();
  });

  it('disables the job tabs while the job lookup runs', async () => {
    mockLineage({ job: { jobName: undefined, workspace: 'ws', isLoading: true } });
    renderPage('/workspaces/ws/fine-tune/models/my-model');

    expect(await screen.findByRole('tab', { name: 'Training' })).toBeDisabled();
    expect(screen.getByRole('tab', { name: 'Logs' })).toBeDisabled();
  });

  it('drops the job tabs when no job is found, falling back to Details', async () => {
    mockLineage({ job: { jobName: undefined, workspace: 'ws', isLoading: false } });
    renderPage('/workspaces/ws/fine-tune/models/my-model/training');

    expect(await screen.findByText('details content')).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Training' })).not.toBeInTheDocument();
  });

  it("lists only this model's evaluations, with a run action that pre-selects it", async () => {
    mockLineage();
    renderPage('/workspaces/ws/fine-tune/models/my-model/evaluations');

    expect(await screen.findByText('evaluations for my-model')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Run Evaluation' })).toHaveAttribute(
      'href',
      '/workspaces/ws/evaluation/new?model=ws%2Fmy-model'
    );
  });

  it("lists an adapter's own evaluations, not its base model's", async () => {
    mockLineage({ adapter });
    renderPage('/workspaces/ws/fine-tune/models/my-model/adapters/my-adapter/evaluations');

    // The composite the run actually stores in `spec.target.name` -- matching on
    // the adapter's bare name would list nothing while looking correct.
    expect(
      await screen.findByText('evaluations for my-model&adapters/ws/my-adapter')
    ).toBeInTheDocument();
  });

  it('pre-selects the adapter when running an evaluation from its page', async () => {
    mockLineage({ adapter });
    renderPage('/workspaces/ws/fine-tune/models/my-model/adapters/my-adapter/evaluations');

    expect(await screen.findByRole('link', { name: 'Run Evaluation' })).toHaveAttribute(
      'href',
      '/workspaces/ws/evaluation/new?model=ws%2Fmy-model%3A%3Amy-adapter'
    );
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

  describe('under the Model Catalog', () => {
    const baseModel = { name: 'my-model', workspace: 'ws' } as ModelEntity;

    it('sits under the Model Catalog breadcrumb', async () => {
      mockLineage({
        model: baseModel,
        job: { jobName: undefined, workspace: 'ws', isLoading: false },
      });
      renderPage('/workspaces/ws/base-models/my-model');

      await screen.findByTestId('nv-page-header-heading');
      expect(useBreadcrumbs).toHaveBeenLastCalledWith({
        items: [
          { href: '/workspaces/ws/base-models', slotLabel: 'Model Catalog' },
          { slotLabel: 'my-model' },
        ],
      });
    });

    it('offers delete on Details while no provider serves the model', async () => {
      mockLineage({
        model: baseModel,
        job: { jobName: undefined, workspace: 'ws', isLoading: false },
      });
      renderPage('/workspaces/ws/base-models/my-model');

      expect(
        await screen.findByRole('button', { name: 'details actions with delete' })
      ).toBeInTheDocument();
    });

    it('withholds delete while a provider serves the model', async () => {
      mockLineage({
        model: { ...baseModel, model_providers: ['ws/provider'] } as ModelEntity,
        job: { jobName: undefined, workspace: 'ws', isLoading: false },
      });
      renderPage('/workspaces/ws/base-models/my-model');

      expect(await screen.findByRole('button', { name: 'details actions' })).toBeInTheDocument();
    });

    it('opens the tab named in the catalog URL', async () => {
      mockLineage({ model: baseModel });
      renderPage('/workspaces/ws/base-models/my-model/deployment');

      expect(await screen.findByText('deployment content')).toBeInTheDocument();
    });

    it('looks the model up in the default workspace too', async () => {
      mockLineage({ model: baseModel });
      renderPage('/workspaces/ws/base-models/my-model');

      await screen.findByTestId('nv-page-header-heading');
      expect(useModelLineage).toHaveBeenLastCalledWith('ws', 'my-model', undefined, {
        fallbackWorkspace: 'default',
      });
    });
  });

  it('sits under the Fine-tuning breadcrumb, without the catalog actions', async () => {
    mockLineage();
    renderPage('/workspaces/ws/fine-tune/models/my-model');

    await screen.findByTestId('nv-page-header-heading');
    expect(useBreadcrumbs).toHaveBeenLastCalledWith({
      items: [
        { href: '/workspaces/ws/fine-tune', slotLabel: 'Fine-tuning' },
        { slotLabel: 'my-model' },
      ],
    });
    expect(screen.queryByRole('button', { name: /details actions/ })).not.toBeInTheDocument();
  });
});
