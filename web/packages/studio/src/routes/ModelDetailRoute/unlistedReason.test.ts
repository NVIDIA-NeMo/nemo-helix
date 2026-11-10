// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getUnlistedReason } from '@studio/routes/ModelDetailRoute/unlistedReason';

describe('getUnlistedReason', () => {
  it('names a provider that dropped the model', () => {
    expect(
      getUnlistedReason({ missingProviders: [], unlistingProviders: ['default/nvidia-build'] })
    ).toBe('nvidia-build no longer lists this model');
  });

  it('names deleted providers', () => {
    expect(
      getUnlistedReason({ missingProviders: ['default/a', 'default/b'], unlistingProviders: [] })
    ).toBe('a, b no longer exist');
  });

  it('gives both reasons when both apply', () => {
    expect(
      getUnlistedReason({
        missingProviders: ['default/build'],
        unlistingProviders: ['default/nvidia-build'],
      })
    ).toBe('nvidia-build no longer lists this model; build no longer exists');
  });
});
