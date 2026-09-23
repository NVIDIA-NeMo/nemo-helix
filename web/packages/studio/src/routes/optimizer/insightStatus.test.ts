// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { insightActions } from '@studio/routes/optimizer/insightStatus';

describe('insightStatus', () => {
  it('only offers transitions accepted by the optimizer API', () => {
    expect(insightActions('open')).toEqual([
      { label: 'Delete', target: 'deleted', kind: 'secondary' },
      { label: 'Resolve', target: 'resolved', kind: 'primary' },
    ]);
    expect(insightActions('resolved')).toEqual([
      { label: 'Reopen', target: 'open', kind: 'primary', color: 'brand' },
    ]);
    expect(insightActions('deleted')).toEqual([
      { label: 'Reopen', target: 'open', kind: 'primary', color: 'brand' },
    ]);
  });
});
