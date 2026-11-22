// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getJobDetailRoute } from '@studio/components/dataViews/JobsDataView/utils';

describe('getJobDetailRoute', () => {
  it('opens Data Designer create jobs on the Data Designer job page', () => {
    expect(
      getJobDetailRoute({ name: 'dd-job-1', source: 'nemo-data-designer-plugin' }, 'default')
    ).toBe('/workspaces/default/data-designer/dd-job-1');
  });

  it.each([
    'data-designer.retrieval-generate',
    'data-designer.retrieval-prepare',
    'data-designer.retrieval-run',
  ])('opens %s jobs on the generic job page', (source) => {
    expect(getJobDetailRoute({ name: 'retrieval-job-1', source }, 'default')).toBe(
      '/workspaces/default/jobs/retrieval-job-1'
    );
  });
});
