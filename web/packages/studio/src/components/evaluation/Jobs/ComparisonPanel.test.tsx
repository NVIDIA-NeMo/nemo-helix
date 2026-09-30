// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/evaluator/schema';
import { ComparisonPanel } from '@studio/components/evaluation/Jobs/ComparisonPanel';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { metricEvaluationJob1 } from '@studio/mocks/evaluation/v1/evaluations';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { renderRoute } from '@studio/tests/util/render';
import { screen, waitFor } from '@testing-library/react';

const workspace = workspace1.name!;

describe('ComparisonPanel', () => {
  beforeEach(() => {
    mockUseParams({
      [ROUTE_PARAMS.workspace]: workspace,
      id: metricEvaluationJob1.id,
    });
  });

  describe('Loading State', () => {
    it('should show panel heading while fetching', async () => {
      renderRoute(<ComparisonPanel job={metricEvaluationJob1} workspace={workspace} jobName="" />);

      expect(screen.getByText('Detailed Metrics')).toBeInTheDocument();
    });
  });

  describe('Pending Status', () => {
    it('should render spinner with message', async () => {
      const pendingJob = {
        ...metricEvaluationJob1,
        status: HelixJobStatus.active,
      };

      renderRoute(<ComparisonPanel job={pendingJob} workspace={workspace} jobName="" />);

      await waitFor(() => {
        expect(screen.getByText(/evaluation in progress/i)).toBeInTheDocument();
      });
    });
  });

  describe('Failed Status', () => {
    it('should render error message', async () => {
      const failedJob = {
        ...metricEvaluationJob1,
        status: HelixJobStatus.error,
      };

      renderRoute(<ComparisonPanel job={failedJob} workspace={workspace} jobName="" />);

      await waitFor(() => {
        expect(screen.getByText('Job Failed')).toBeInTheDocument();
      });

      expect(
        screen.getByText('The evaluation job failed. No comparison data available.')
      ).toBeInTheDocument();
    });
  });

  describe('Completed Status', () => {
    it('should render the panel heading for completed jobs', async () => {
      renderRoute(<ComparisonPanel job={metricEvaluationJob1} workspace={workspace} jobName="" />);

      expect(screen.getByText('Detailed Metrics')).toBeInTheDocument();
    });
  });
});
