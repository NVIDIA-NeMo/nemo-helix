// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { asRecord } from '@studio/util/guards';

describe('asRecord', () => {
  it('returns keyed objects as-is', () => {
    const value = { status: 404, response: { data: { detail: 'nope' } } };
    expect(asRecord(value)).toBe(value);
  });

  it.each([null, undefined, 'text', 0, true, [], [{ msg: 'x' }]])('rejects %j', (value) => {
    expect(asRecord(value)).toBeUndefined();
  });
});
