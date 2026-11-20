// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { invalidateSessionDeletion } from '@studio/api/intake/invalidateSessionDeletion';
import { QueryClient } from '@tanstack/react-query';

it('invalidates session, trace, span, group and metric queries only in the affected workspace', async () => {
  const client = new QueryClient();
  const affected = [
    'sessions/a',
    'traces',
    'traces/a',
    'spans',
    'spans/groups',
    'traces/metrics',
  ].map((path) => [`/apis/intake/v2/workspaces/${DEFAULT_WORKSPACE}/${path}`]);
  const unaffected = [['/apis/intake/v2/workspaces/another/traces'], ['unrelated']];
  [...affected, ...unaffected].forEach((key) => client.setQueryData(key, {}));
  await invalidateSessionDeletion(client, DEFAULT_WORKSPACE);
  affected.forEach((key) => expect(client.getQueryState(key)?.isInvalidated).toBe(true));
  unaffected.forEach((key) => expect(client.getQueryState(key)?.isInvalidated).toBe(false));
  client.clear();
});
