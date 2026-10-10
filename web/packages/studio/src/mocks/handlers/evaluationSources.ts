// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import { getListExperimentsQueryKey } from '@nemo/sdk/generated/platform/experiments';
import type { EvaluationResponse, ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { http, HttpResponse } from 'msw';

const byNameFilter = <T extends { name: string }>(request: Request, items: T[]): T[] => {
  const name = new URL(request.url).searchParams.get('filter[name]');
  return name ? items.filter((item) => item.name === name) : items;
};

/** Intake experiment + evaluation lists. A `filter[name]` request is a name-conflict probe and
 *  gets only the exact match; anything else gets the whole list. */
export const evaluationSourcesHandlers = ({
  experiments = [],
  evaluations = [],
}: {
  experiments?: ExperimentResponse[];
  evaluations?: EvaluationResponse[];
}) => [
  http.get(mockApiUrl(getListExperimentsQueryKey, ':workspace'), ({ request }) =>
    HttpResponse.json({ data: byNameFilter(request, experiments) })
  ),
  http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), ({ request }) =>
    HttpResponse.json({ data: byNameFilter(request, evaluations) })
  ),
];
