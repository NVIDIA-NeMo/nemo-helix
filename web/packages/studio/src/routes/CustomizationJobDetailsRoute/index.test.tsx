// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import { HelixJobResponse, HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import {
  customizationJob1,
  failedGrpoCustomizationJob,
} from '@studio/mocks/customizer/customization-jobs';
import { dataset } from '@studio/mocks/datasets';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { CustomizationJobDetailsRoute } from '@studio/routes/CustomizationJobDetailsRoute';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { mockUseNavigate, mockUseParams } from '@studio/tests/util/mockUseParams';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { getBaseModel } from '@studio/util/customizations';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

describe('CustomizationJobDetailsRoute', () => {
  beforeEach(() => {
    mockUseNavigate();
    mockUseParams({
      [ROUTE_PARAMS.workspace]: workspace1.name,
      [ROUTE_PARAMS.customizationJobName]: customizationJob1.name,
    });
  });

  afterEach(() => {
    server.resetHandlers();
  });
  it('should render customization details', async () => {
    render(
      <TestProviders>
        <CustomizationJobDetailsRoute />
      </TestProviders>
    );

    // Use longer timeout
    expect(
      (await screen.findAllByText(customizationJob1.id!, {}, { timeout: XL_SELECTOR_TIMEOUT }))
        .length
    ).toBeGreaterThan(0);
    expect(
      (
        await screen.findAllByText(
          getEntityReference(dataset),
          {},
          { timeout: XL_SELECTOR_TIMEOUT }
        )
      ).length
    ).toBeGreaterThan(0);
    expect(screen.queryByTestId('customization-error-banner')).not.toBeInTheDocument();
  });

  it('names the job in the page header with its type, base model, and created date', async () => {
    render(
      <TestProviders>
        <CustomizationJobDetailsRoute />
      </TestProviders>
    );

    expect(
      await screen.findByText('LoRA', {}, { timeout: XL_SELECTOR_TIMEOUT })
    ).toBeInTheDocument();

    expect(screen.getByText(customizationJob1.name!)).toBeInTheDocument();
    expect(screen.getAllByText(getBaseModel(customizationJob1)).length).toBeGreaterThan(0);
    expect(
      screen.getByText(`created ${formatAbsoluteTimestamp(customizationJob1.created_at!)}`)
    ).toBeInTheDocument();
  });

  it('keeps polling an active job after switching away from the Overview tab', async () => {
    let jobRequests = 0;
    server.use(
      http.get<never, never, HelixJobResponse>(
        `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs/:name`,
        () => {
          jobRequests += 1;
          return HttpResponse.json({
            ...customizationJob1,
            status: HelixJobStatus.active,
          } as unknown as HelixJobResponse);
        }
      )
    );

    const user = userEvent.setup();
    render(
      <TestProviders>
        <CustomizationJobDetailsRoute />
      </TestProviders>
    );

    await user.click(await screen.findByRole('tab', { name: /Logs/ }));
    expect(screen.queryByText('Run configuration')).not.toBeInTheDocument();

    const requestsAfterSwitch = jobRequests;
    await waitFor(() => expect(jobRequests).toBeGreaterThan(requestsAfterSwitch), {
      timeout: XL_SELECTOR_TIMEOUT,
    });
  });

  it('shows the job logs in their own tab', async () => {
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs/:name/logs`, () =>
        HttpResponse.json({
          data: [
            {
              timestamp: '2025-10-24T15:13:17Z',
              job: customizationJob1.name,
              job_step: 'training',
              job_task: 'main',
              message: 'The training job is pending',
            },
          ],
          total: 1,
          next_page: '',
          prev_page: '',
        })
      )
    );

    const user = userEvent.setup();
    render(
      <TestProviders>
        <CustomizationJobDetailsRoute />
      </TestProviders>
    );

    await user.click(await screen.findByRole('tab', { name: /Logs/ }));

    await waitFor(
      () => {
        expect(
          screen.getByText((content) => content.includes('The training job is pending'))
        ).toBeInTheDocument();
      },
      { timeout: XL_SELECTOR_TIMEOUT }
    );
  });

  describe('a failed job', () => {
    beforeEach(() => {
      mockUseNavigate();
      mockUseParams({
        [ROUTE_PARAMS.workspace]: workspace1.name,
        [ROUTE_PARAMS.customizationJobName]: failedGrpoCustomizationJob.name,
      });
    });

    it('shows the mapped cause from the failing task, not the generic job-level text', async () => {
      render(
        <TestProviders>
          <CustomizationJobDetailsRoute />
        </TestProviders>
      );

      const banner = await screen.findByTestId('customization-error-banner', undefined, {
        timeout: XL_SELECTOR_TIMEOUT,
      });

      expect(banner).toHaveTextContent(/ran out of GPU memory/);
      expect(banner).not.toHaveTextContent(/One or more tasks are in error state/);
    });

    it('names the failing pipeline step and the mapped error type', async () => {
      render(
        <TestProviders>
          <CustomizationJobDetailsRoute />
        </TestProviders>
      );

      const banner = await screen.findByTestId('customization-error-banner', undefined, {
        timeout: XL_SELECTOR_TIMEOUT,
      });

      expect(banner).toHaveTextContent('Failed during GRPO training');
      expect(banner).toHaveTextContent('CudaError');
    });

    it('opens the logs tab from the banner', async () => {
      const user = userEvent.setup();

      render(
        <TestProviders>
          <CustomizationJobDetailsRoute />
        </TestProviders>
      );

      await user.click(
        await screen.findByRole(
          'button',
          { name: /View GRPO training logs/ },
          { timeout: XL_SELECTOR_TIMEOUT }
        )
      );

      expect(await screen.findByRole('tab', { name: /Logs/, selected: true })).toBeInTheDocument();
    });
  });
});
