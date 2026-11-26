// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from '@storybook/react';
import { ROUTES } from '@studio/constants/routes';
import {
  SAFE_SYNTHESIZER_REPORT_DATA_URL,
  safeSynthesizerHandlers,
} from '@studio/mocks/handlers/safeSynthesizer';
// eslint-disable-next-line import/extensions -- library output kept as plain JSON so it can be regenerated as is
import timeSeriesReportData from '@studio/mocks/safe-synthesizer/evaluation-report-data-time-series.json';
import { BreadcrumbsProvider } from '@studio/providers/breadcrumbs/BreadcrumbsProvider';
import { GenerateJobReportRoute } from '@studio/routes/SafeSynthesizerJobReportRoute';
import { http, HttpResponse } from 'msw';
import { Suspense } from 'react';
import { Route, Routes } from 'react-router';

const REPORT_PATH = '/workspaces/default/safe-synthesizer/job/customers-job/report';

const meta = {
  title: 'Safe Synthesizer/Job Report',
  component: GenerateJobReportRoute,
  decorators: [
    (Story) => (
      <BreadcrumbsProvider>
        <Suspense fallback={null}>
          <Routes>
            <Route path={ROUTES.workspace.safeSynthesizerJobReport} element={<Story />} />
          </Routes>
        </Suspense>
      </BreadcrumbsProvider>
    ),
  ],
  parameters: {
    layout: 'fullscreen',
    msw: { handlers: safeSynthesizerHandlers },
    router: { initialPath: REPORT_PATH },
  },
} satisfies Meta<typeof GenerateJobReportRoute>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Overview: Story = {};

export const SyntheticQuality: Story = {
  parameters: { router: { initialPath: `${REPORT_PATH}?tab=synthetic-quality` } },
};

export const DataPrivacy: Story = {
  parameters: { router: { initialPath: `${REPORT_PATH}?tab=data-privacy` } },
};

export const TrainingData: Story = {
  parameters: { router: { initialPath: `${REPORT_PATH}?tab=training-data` } },
};

export const TimeSeries: Story = {
  parameters: {
    router: { initialPath: `${REPORT_PATH}?tab=synthetic-quality` },
    msw: {
      handlers: [
        http.get(SAFE_SYNTHESIZER_REPORT_DATA_URL, () => HttpResponse.json(timeSeriesReportData)),
        ...safeSynthesizerHandlers,
      ],
    },
  },
};

export const WithoutReportData: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get(SAFE_SYNTHESIZER_REPORT_DATA_URL, () =>
          HttpResponse.json({ detail: 'Result not found' }, { status: 404 })
        ),
        ...safeSynthesizerHandlers,
      ],
    },
  },
};
