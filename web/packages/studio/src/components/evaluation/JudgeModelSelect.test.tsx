// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import type { FC, PropsWithChildren } from 'react';
import { FormProvider, useForm } from 'react-hook-form';
import { MemoryRouter } from 'react-router';

const MODELS_URL = `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models`;
const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;

/** Deployed and servable: it carries an endpoint, so chat is available. */
const deployedModel = {
  id: 'm1',
  name: 'deployed-model',
  workspace: DEFAULT_WORKSPACE,
  created_at: '2026-10-01T00:00:00Z',
  updated_at: '2026-10-01T00:00:00Z',
  api_endpoint: { url: 'https://example.invalid/v1' },
} as unknown as ModelEntity;

const modelsPage = (models: ModelEntity[]) => ({
  data: models,
  pagination: {
    page: 1,
    page_size: 25,
    current_page_size: models.length,
    total_pages: 1,
    total_results: models.length,
  },
});

const inFlightJob = {
  id: 'j1',
  name: 'automodel-003',
  workspace: DEFAULT_WORKSPACE,
  source: 'customization',
  status: 'active',
  created_at: '2026-10-06T00:00:00Z',
  spec: { model: 'default/nemotron', output: { name: 'bird-sql-003' } },
};

const jobsResponse = (jobs: unknown[]) => ({
  data: jobs,
  pagination: { page: 1, page_size: 50, total_results: jobs.length, total_pages: 1 },
});

const FormWrapper: FC<PropsWithChildren> = ({ children }) => {
  const methods = useForm({ defaultValues: { model: '' } });
  return (
    <TestProviders>
      <MemoryRouter>
        <FormProvider {...methods}>{children}</FormProvider>
      </MemoryRouter>
    </TestProviders>
  );
};

const openDropdown = async () => {
  const user = userEvent.setup();
  await user.click(await screen.findByTestId('model-select-v2-trigger'));
  return user;
};

describe('JudgeModelSelect', () => {
  beforeEach(() => {
    mockUseParams({ [ROUTE_PARAMS.workspace]: DEFAULT_WORKSPACE });
    server.use(
      http.get(MODELS_URL, () => HttpResponse.json(modelsPage([deployedModel]))),
      http.get(JOBS_URL, () => HttpResponse.json(jobsResponse([inFlightJob])))
    );
  });

  describe('by default', () => {
    it('lists registered models only, leaving runs still in flight out', async () => {
      render(
        <FormWrapper>
          <JudgeModelSelect formFieldName="model" />
        </FormWrapper>
      );

      await openDropdown();

      expect(await screen.findByText('deployed-model')).toBeInTheDocument();
      expect(screen.queryByText('bird-sql-003')).not.toBeInTheDocument();
    });
  });

  describe('with showUnavailable', () => {
    it('leaves a registered, servable model selectable', async () => {
      render(
        <FormWrapper>
          <JudgeModelSelect formFieldName="model" showUnavailable />
        </FormWrapper>
      );

      await openDropdown();

      expect(await screen.findByText('deployed-model')).toBeInTheDocument();
      const selectable = screen.getAllByTestId('model-dropdown-item');
      expect(selectable.map((item) => item.textContent)).toContain('deployed-model');
    });

    it('shows a fine-tuning run that has no model entity yet, with its state', async () => {
      render(
        <FormWrapper>
          <JudgeModelSelect formFieldName="model" showUnavailable />
        </FormWrapper>
      );

      await openDropdown();

      expect(await screen.findByText('bird-sql-003')).toBeInTheDocument();
      expect(screen.getByText('Fine-tuning active — not ready to evaluate')).toBeInTheDocument();
    });

    it('matches an in-flight run on its job name, not only its future model name', async () => {
      render(
        <FormWrapper>
          <JudgeModelSelect formFieldName="model" showUnavailable />
        </FormWrapper>
      );

      const user = await openDropdown();
      await screen.findByText('bird-sql-003');

      await user.type(screen.getByTestId('model-select-v2-filter'), 'automodel-003');

      expect(await screen.findByText('bird-sql-003')).toBeInTheDocument();
    });

    it('drops the placeholder once the run has registered its model', async () => {
      server.use(
        http.get(MODELS_URL, () =>
          HttpResponse.json(
            modelsPage([{ ...deployedModel, id: 'm3', name: 'bird-sql-003' } as ModelEntity])
          )
        )
      );

      render(
        <FormWrapper>
          <JudgeModelSelect formFieldName="model" showUnavailable />
        </FormWrapper>
      );

      await openDropdown();

      await waitFor(() => expect(screen.getAllByText('bird-sql-003')).toHaveLength(1));
      expect(
        screen.queryByText('Fine-tuning active — not ready to evaluate')
      ).not.toBeInTheDocument();
    });
  });
});
