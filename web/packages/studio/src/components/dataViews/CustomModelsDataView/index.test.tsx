// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { ModelEntitysPage } from '@nemo/sdk/generated/platform/schema';
import { CustomModelsDataView } from '@studio/components/dataViews/CustomModelsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import {
  entityStoreBaseModel1,
  entityStoreCustomizedModel1,
  entityStorePromptTunedModel1,
} from '@studio/mocks/entity-store/models';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { getCustomizationJobListRoute } from '@studio/routes/utils';
import { renderRoute } from '@studio/tests/util/render';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const MODELS_URL = `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models`;
const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;
const ADAPTER_DELETE_URL = `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models/:modelName/adapters/:adapter`;

const modelsPage: ModelEntitysPage = {
  data: [entityStoreBaseModel1, entityStorePromptTunedModel1, entityStoreCustomizedModel1],
  pagination: {
    page: 1,
    page_size: 30,
    current_page_size: 30,
    total_pages: 1,
    total_results: 3,
  },
};

const emptyModelsPage: ModelEntitysPage = {
  data: [],
  pagination: {
    page: 1,
    page_size: 30,
    current_page_size: 30,
    total_pages: 0,
    total_results: 0,
  },
};

const renderComponent = () =>
  renderRoute(<CustomModelsDataView workspace={workspace1.workspace} />, {
    history: getCustomizationJobListRoute(workspace1.workspace),
    routes: [
      {
        path: ROUTES.workspace.customizationJobList,
        element: <CustomModelsDataView workspace={workspace1.workspace} />,
      },
      {
        path: ROUTES.workspace.customizationJobDetails,
        element: <div>Job details page</div>,
      },
    ],
  });

const jobsPage = (jobs: { id: string; name: string }[]) => ({
  data: jobs,
  pagination: { page: 1, page_size: 1, total_results: jobs.length, total_pages: 1 },
});

/** Opens the row-actions menu of the row whose name cell reads `rowName`. */
const openRowActions = async (user: ReturnType<typeof userEvent.setup>, rowName: string) => {
  await screen.findByText(rowName);
  const row = screen.getAllByRole('row').find((r) => within(r).queryByText(rowName));
  if (!row) throw new Error(`No table row for ${rowName}`);
  await user.click(within(row).getByRole('button', { name: /actions/i }));
};

vi.mock('@nemo/common/src/components/DataView/StudioDataView', async () => {
  const { studioDataViewMock } = await import('@studio/tests/util');
  return studioDataViewMock();
});

describe('CustomModelsDataView', () => {
  it('renders the empty state when no models are available', async () => {
    server.use(http.get(MODELS_URL, () => HttpResponse.json(emptyModelsPage)));

    renderComponent();

    expect(await screen.findByText(ENTITY_EMPTY_STATES.customModels.heading)).toBeInTheDocument();
  });

  it('renders the error state when there is an error fetching models', async () => {
    server.use(http.get(MODELS_URL, () => HttpResponse.error()));

    renderComponent();

    expect(await screen.findByText('Loading Error')).toBeInTheDocument();
    expect(await screen.findByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('renders the expected columns and model data', async () => {
    server.use(http.get(MODELS_URL, () => HttpResponse.json(modelsPage)));

    renderComponent();

    await waitFor(() => expect(screen.queryByTestId('spinner')).not.toBeInTheDocument());

    const columns = ['Name', 'Base Model', 'Created'];
    for (const columnHeader of columns) {
      expect(await screen.findByRole('columnheader', { name: columnHeader })).toBeInTheDocument();
    }

    expect(screen.getAllByText(entityStoreBaseModel1.name).length).toBeGreaterThan(0);
    expect(screen.getAllByRole('row').length).toBeGreaterThan(0);
  });

  it('deletes an adapter via the correct API endpoint', async () => {
    const user = userEvent.setup();
    const adapterDeleteHandler = vi.fn();
    const adapterName = entityStoreCustomizedModel1.adapters![0].name;

    server.use(
      http.get(MODELS_URL, () => HttpResponse.json(modelsPage)),
      http.delete(ADAPTER_DELETE_URL, ({ params }) => {
        adapterDeleteHandler(params);
        return new HttpResponse(null, { status: 204 });
      })
    );

    renderComponent();

    await waitFor(() => expect(screen.queryByTestId('spinner')).not.toBeInTheDocument());

    // Sub-rows are expanded by default — find the adapter and open its actions menu
    await screen.findByText(adapterName);
    const actionButtons = screen.getAllByRole('button', { name: /actions/i });
    const adapterActionButton = actionButtons[actionButtons.length - 1];
    await user.click(adapterActionButton);

    // Click "Delete Adapter" in the menu
    await user.click(await screen.findByRole('menuitem', { name: 'Delete Adapter' }));

    // Confirm deletion in the modal
    expect(await screen.findByText('Delete Adapter')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => {
      expect(adapterDeleteHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          modelName: entityStoreCustomizedModel1.name,
          adapter: adapterName,
        })
      );
    });
  });

  describe('Job details action', () => {
    it.each([
      { kind: 'model', outputName: entityStoreCustomizedModel1.name },
      { kind: 'adapter', outputName: entityStoreCustomizedModel1.adapters![0].name },
    ])('opens the fine-tuning job that produced the $kind', async ({ outputName }) => {
      const user = userEvent.setup();
      let capturedUrl: URL | undefined;
      server.use(
        http.get(MODELS_URL, () => HttpResponse.json(modelsPage)),
        http.get(JOBS_URL, ({ request }) => {
          capturedUrl = new URL(request.url);
          return HttpResponse.json(jobsPage([{ id: 'j1', name: 'automodel-2a4c86bf' }]));
        })
      );

      renderComponent();
      await openRowActions(user, outputName);
      await user.click(await screen.findByRole('menuitem', { name: 'Job details' }));

      expect(await screen.findByText('Job details page')).toBeInTheDocument();
      expect(capturedUrl?.searchParams.get('filter[source]')).toBe('customization');
      expect(capturedUrl?.searchParams.get('filter[spec.output.name]')).toBe(outputName);
    });

    it('reports when no fine-tuning job produced the model', async () => {
      const user = userEvent.setup();
      server.use(
        http.get(MODELS_URL, () => HttpResponse.json(modelsPage)),
        http.get(JOBS_URL, () => HttpResponse.json(jobsPage([])))
      );

      renderComponent();
      await openRowActions(user, entityStoreCustomizedModel1.name);
      await user.click(await screen.findByRole('menuitem', { name: 'Job details' }));

      expect(
        await screen.findByText(`No fine-tuning job found for ${entityStoreCustomizedModel1.name}.`)
      ).toBeInTheDocument();
      expect(screen.queryByText('Job details page')).not.toBeInTheDocument();
    });
  });
});
