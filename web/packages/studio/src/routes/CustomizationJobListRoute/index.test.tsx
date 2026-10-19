// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { CustomizationJobListRoute } from '@studio/routes/CustomizationJobListRoute';
import { renderRoute } from '@studio/tests/util/render';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const flags = vi.hoisted(() => ({ modelDetails: false }));
vi.mock('@studio/constants/environment', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/constants/environment')>()),
  get MODEL_DETAILS_ENABLED() {
    return flags.modelDetails;
  },
}));

const MODELS_URL = `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models`;
const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;

const adapter = {
  name: 'grumpy-purple-gorilla',
  workspace: 'ws',
  fileset: 'ws/grumpy-purple-gorilla',
  finetuning_type: 'lora',
  created_at: '2025-01-02T00:00:00Z',
};

/** Fine-tuned itself *and* carrying an adapter — both runs exist and could be linked. */
const fineTunedBase = {
  id: 'm1',
  name: 'qwen3-0.6b-sft',
  workspace: 'ws',
  base_model: 'qwen/qwen3-0.6b',
  finetuning_type: 'all_weights',
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
  adapters: [adapter],
};

const modelsPage = {
  data: [fineTunedBase],
  pagination: { page: 1, page_size: 30, current_page_size: 1, total_pages: 1, total_results: 1 },
};

/** Each entity resolves to its own run, keyed on the output name in the request. */
const jobsByOutputName: Record<string, string> = {
  'qwen3-0.6b-sft': 'sft-job-123',
  'grumpy-purple-gorilla': 'lora-job-456',
};

vi.mock('@nemo/common/src/components/DataView/StudioDataView', async () => {
  const { studioDataViewMock } = await import('@studio/tests/util');
  return studioDataViewMock();
});

const renderRouteUnderTest = () =>
  renderRoute(<CustomizationJobListRoute />, {
    history: '/workspaces/ws/fine-tune',
    routes: [
      { path: ROUTES.workspace.customizationJobList, element: <CustomizationJobListRoute /> },
      { path: ROUTES.workspace.modelDetail, element: <div>model page</div> },
      { path: ROUTES.workspace.modelAdapterDetail, element: <div>adapter page</div> },
    ],
  });

const jobLinkHrefs = () =>
  screen.queryAllByRole('link', { name: 'View Job Details' }).map((l) => l.getAttribute('href'));

describe('CustomizationJobListRoute job links', () => {
  beforeEach(() => {
    server.use(
      http.get(MODELS_URL, () => HttpResponse.json(modelsPage)),
      http.get(JOBS_URL, ({ request }) => {
        const outputName = new URL(request.url).searchParams.get('filter[spec.output.name]') ?? '';
        const jobName = jobsByOutputName[outputName];
        return HttpResponse.json({
          data: jobName ? [{ id: `id-${jobName}`, name: jobName }] : [],
          pagination: { page: 1, page_size: 1, total_results: jobName ? 1 : 0, total_pages: 1 },
        });
      })
    );
  });

  it("shows only the adapter's run when an adapter row is opened", async () => {
    const user = userEvent.setup();
    renderRouteUnderTest();

    await user.click(await screen.findByText(adapter.name));

    // The base is itself a fine-tune with its own run, so both links are available. The
    // user asked about the adapter; offering the base's run alongside invites attributing
    // one to the other.
    await waitFor(() => expect(jobLinkHrefs()).toHaveLength(1));
    expect(jobLinkHrefs()[0]).toEqual(expect.stringContaining('lora-job-456'));
    expect(jobLinkHrefs()[0]).not.toEqual(expect.stringContaining('sft-job-123'));
  });

  it("shows the model's own run when a model row is opened", async () => {
    const user = userEvent.setup();
    renderRouteUnderTest();

    await user.click(await screen.findByText(fineTunedBase.name));

    await waitFor(() => expect(jobLinkHrefs()).toHaveLength(1));
    expect(jobLinkHrefs()[0]).toEqual(expect.stringContaining('sft-job-123'));
  });
});

describe('CustomizationJobListRoute with the model details flag', () => {
  beforeEach(() => {
    flags.modelDetails = true;
    server.use(http.get(MODELS_URL, () => HttpResponse.json(modelsPage)));
  });

  afterEach(() => {
    flags.modelDetails = false;
  });

  it('opens the model page instead of the side panel', async () => {
    const user = userEvent.setup();
    renderRouteUnderTest();

    await user.click(await screen.findByText(fineTunedBase.name));

    expect(await screen.findByText('model page')).toBeInTheDocument();
  });

  it('opens the adapter page for an adapter row', async () => {
    const user = userEvent.setup();
    renderRouteUnderTest();

    await user.click(await screen.findByText(adapter.name));

    expect(await screen.findByText('adapter page')).toBeInTheDocument();
  });
});
