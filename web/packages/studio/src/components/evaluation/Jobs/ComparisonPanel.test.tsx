// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getEvaluatorListEvaluateJobResultsQueryKey } from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import { PlatformJobStatus } from '@nemo/sdk/generated/evaluator/schema';
import { ComparisonPanel } from '@studio/components/evaluation/Jobs/ComparisonPanel';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { metricEvaluationJob1 } from '@studio/mocks/evaluation/v1/evaluations';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { renderRoute } from '@studio/tests/util/render';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

const workspace = workspace1.name!;
const jobName = metricEvaluationJob1.name;

describe('ComparisonPanel', () => {
  beforeEach(() => {
    mockUseParams({
      [ROUTE_PARAMS.workspace]: workspace,
      id: metricEvaluationJob1.id,
    });

    // Mock the results list endpoint used by useEvaluatorListEvaluateJobResults
    server.use(
      http.get(
        mockApiUrl(getEvaluatorListEvaluateJobResultsQueryKey, ':workspace', ':jobName'),
        () => {
          return HttpResponse.json({ data: [], pagination: {} });
        }
      )
    );
  });

  describe('Loading State', () => {
    it('should show panel heading while fetching', async () => {
      renderRoute(
        <ComparisonPanel job={metricEvaluationJob1} workspace={workspace} jobName={jobName} />
      );

      expect(screen.getByText('Detailed Metrics')).toBeInTheDocument();
    });
  });

  describe('Pending Status', () => {
    it('should render spinner with message', async () => {
      const pendingJob = {
        ...metricEvaluationJob1,
        status: PlatformJobStatus.active,
      };

      renderRoute(<ComparisonPanel job={pendingJob} workspace={workspace} jobName={jobName} />);

      await waitFor(() => {
        expect(screen.getByText(/evaluation in progress/i)).toBeInTheDocument();
      });
    });
  });

  describe('Failed Status', () => {
    it('should render error message', async () => {
      const failedJob = {
        ...metricEvaluationJob1,
        status: PlatformJobStatus.error,
      };

      renderRoute(<ComparisonPanel job={failedJob} workspace={workspace} jobName={jobName} />);

      await waitFor(() => {
        expect(screen.getByText('Job Failed')).toBeInTheDocument();
      });

      expect(
        screen.getByText('The evaluation job failed. No comparison data available.')
      ).toBeInTheDocument();
    });
  });

  describe('Completed Status', () => {
    beforeEach(() => {
      server.use(
        http.get(
          mockApiUrl(getEvaluatorListEvaluateJobResultsQueryKey, ':workspace', ':jobName'),
          () => {
            return HttpResponse.json({
              download_url: 'http://localhost/mock-scores.json',
            });
          }
        )
      );
    });

    it('should render the panel heading for completed jobs', async () => {
      renderRoute(
        <ComparisonPanel job={metricEvaluationJob1} workspace={workspace} jobName={jobName} />
      );

      expect(screen.getByText('Detailed Metrics')).toBeInTheDocument();
    });
  });
});
