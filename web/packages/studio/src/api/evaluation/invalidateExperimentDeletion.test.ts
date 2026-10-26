// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  getGetEvaluationQueryKey,
  getListEvaluationsQueryKey,
} from '@nemo/sdk/generated/platform/evaluations';
import {
  getGetExperimentQueryKey,
  getListExperimentsQueryKey,
} from '@nemo/sdk/generated/platform/experiments';
import { invalidateExperimentDeletion } from '@studio/api/evaluation/invalidateExperimentDeletion';
import { QueryClient } from '@tanstack/react-query';

it('invalidates filtered lists, detail records and agent rollups only in the affected workspace', async () => {
  const client = new QueryClient();
  const keys = [
    getListExperimentsQueryKey(DEFAULT_WORKSPACE, { page: 2 }),
    getGetExperimentQueryKey(DEFAULT_WORKSPACE, 'sample'),
    getListEvaluationsQueryKey(DEFAULT_WORKSPACE, { page: 1 }),
    getGetEvaluationQueryKey(DEFAULT_WORKSPACE, 'result'),
    ['experiments-by-id', DEFAULT_WORKSPACE, ['experiment']],
  ];
  const unaffected = [
    getListExperimentsQueryKey('other-workspace'),
    ['experiments-by-id', 'other-workspace', ['experiment']],
    ['unrelated'],
  ];
  [...keys, ...unaffected].forEach((key) => client.setQueryData(key, {}));
  await invalidateExperimentDeletion(client, DEFAULT_WORKSPACE);
  keys.forEach((key) => expect(client.getQueryState(key)?.isInvalidated).toBe(true));
  unaffected.forEach((key) => expect(client.getQueryState(key)?.isInvalidated).toBe(false));
  client.clear();
});
