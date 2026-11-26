// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getSafeSynthesizerDownloadJobResultEvaluationReportQueryKey,
  getSafeSynthesizerDownloadJobResultQueryKey,
  getSafeSynthesizerDownloadJobResultSummaryQueryKey,
  getSafeSynthesizerGetJobQueryKey,
} from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import type {
  GenerateJob,
  SafeSynthesizerSummary,
} from '@nemo/sdk/generated/safe-synthesizer/schema';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
// eslint-disable-next-line import/extensions -- library output kept as plain JSON so it can be regenerated as is
import reportData from '@studio/mocks/safe-synthesizer/evaluation-report-data.json';
import { EVALUATION_REPORT_DATA_RESULT } from '@studio/routes/SafeSynthesizerJobReportRoute/useEvaluationReportData';
import { SUCCESSFUL_STATUS } from '@studio/util/safeSynthesizer';
import { http, HttpResponse } from 'msw';

export const SAFE_SYNTHESIZER_JOB_URL = mockApiUrl(
  getSafeSynthesizerGetJobQueryKey,
  ':workspace',
  ':name'
);
export const SAFE_SYNTHESIZER_SUMMARY_URL = mockApiUrl(
  getSafeSynthesizerDownloadJobResultSummaryQueryKey,
  ':workspace',
  ':job'
);
export const SAFE_SYNTHESIZER_REPORT_URL = mockApiUrl(
  getSafeSynthesizerDownloadJobResultEvaluationReportQueryKey,
  ':workspace',
  ':job'
);
export const SAFE_SYNTHESIZER_REPORT_DATA_URL = mockApiUrl(
  getSafeSynthesizerDownloadJobResultQueryKey,
  ':workspace',
  ':job',
  EVALUATION_REPORT_DATA_RESULT
);

export const safeSynthesizerJob = (name: string): GenerateJob => ({
  name,
  status: SUCCESSFUL_STATUS,
  spec: { data_source: 'default/customers', config: { privacy: { dp_enabled: false } } },
});

/** Scores from the evaluation-report-data fixture, as the job's summary result reports them. */
export const safeSynthesizerSummary: SafeSynthesizerSummary = {
  synthetic_data_quality_score: reportData.scores.synthetic_quality.score,
  data_privacy_score: reportData.scores.data_privacy.score,
  column_correlation_stability_score: reportData.metrics.column_correlation_stability.score.score,
  deep_structure_stability_score: reportData.metrics.deep_structure_stability.score.score,
  column_distribution_stability_score: reportData.metrics.column_distribution_stability.score.score,
  text_semantic_similarity_score: reportData.metrics.text_semantic_similarity.score.score,
  text_structure_similarity_score: reportData.metrics.text_structure_similarity.score.score,
  membership_inference_protection_score:
    reportData.metrics.membership_inference_protection.score.score,
  attribute_inference_protection_score:
    reportData.metrics.attribute_inference_protection.score.score,
  timing: {},
};

export const safeSynthesizerHandlers = [
  http.get(SAFE_SYNTHESIZER_JOB_URL, ({ params }) =>
    HttpResponse.json(safeSynthesizerJob(String(params.name)))
  ),
  http.get(SAFE_SYNTHESIZER_SUMMARY_URL, () => HttpResponse.json(safeSynthesizerSummary)),
  http.get(SAFE_SYNTHESIZER_REPORT_URL, () =>
    HttpResponse.text('<html><body>Safe Synthesizer Quality &amp; Privacy Report</body></html>', {
      headers: { 'Content-Type': 'text/html' },
    })
  ),
  http.get(SAFE_SYNTHESIZER_REPORT_DATA_URL, () => HttpResponse.json(reportData)),
];
