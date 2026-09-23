// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { DetailsPanel } from '@studio/components/evaluation/Jobs/DetailsPanel';
import { ROUTE_PARAMS, ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { metricEvaluationJob1 } from '@studio/mocks/evaluation/v1/evaluations';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { renderRoute } from '@studio/tests/util/render';
import { screen } from '@testing-library/react';
import { generatePath } from 'react-router';

describe('DetailsPanel', () => {
  beforeEach(() => {
    mockUseParams({
      [ROUTE_PARAMS.workspace]: workspace1.workspace,
      [ROUTE_PARAMS.evaluationJobId]: metricEvaluationJob1.id,
    });
  });
  // Use workspace from workspace1, not name
  const workspace = workspace1.workspace;
  const evaluationJobId = metricEvaluationJob1.id!;
  const testPath = generatePath(ROUTES.workspace.evaluationMetricDetails!, {
    workspace,
    id: evaluationJobId,
  });

  it('renders core details for a completed job', async () => {
    renderRoute(<DetailsPanel evaluationJob={metricEvaluationJob1} />, {
      history: testPath,
      routes: [
        {
          path: ROUTES.workspace.evaluationMetricDetails!,
          element: <DetailsPanel evaluationJob={metricEvaluationJob1} />,
        },
      ],
    });

    // Wait for content to load - Heading and sections
    expect(await screen.findByText('Details')).toBeInTheDocument();

    // Label/value rows
    expect(screen.getByText('Status')).toBeInTheDocument();
    expect(screen.getByText('Created')).toBeInTheDocument();
    expect(screen.getByText('Job ID')).toBeInTheDocument();
    expect(screen.getByText('Model')).toBeInTheDocument();
  });

  it('shows an error banner when the job has failed', async () => {
    const failedJob = {
      ...metricEvaluationJob1,
      status: HelixJobStatus.error,
      error_details: { message: 'The evaluation job failed. Please try again.' },
    };

    renderRoute(<DetailsPanel evaluationJob={failedJob} />, {
      history: testPath,
      routes: [
        {
          path: ROUTES.workspace.evaluationMetricDetails!,
          element: <DetailsPanel evaluationJob={failedJob} />,
        },
      ],
    });

    expect(
      await screen.findByText('The evaluation job failed. Please try again.')
    ).toBeInTheDocument();
  });

  it('renders reload state when evaluationJob is undefined', async () => {
    renderRoute(<DetailsPanel evaluationJob={undefined} />, {
      history: testPath,
      routes: [
        {
          path: ROUTES.workspace.evaluationMetricDetails!,
          element: <DetailsPanel evaluationJob={undefined} />,
        },
      ],
    });

    expect(await screen.findByText('Failed to Load Job')).toBeInTheDocument();
    expect(screen.getByText('Unable to load job. Please try again.')).toBeInTheDocument();

    // Verify reload and launch buttons are present
    expect(screen.getByRole('button', { name: /Reload Job/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Launch Evaluation/i })).toBeInTheDocument();
  });
});
