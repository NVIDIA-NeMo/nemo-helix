// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as featureFlagsModule from '@studio/constants/featureFlags';
import type { FeatureFlags } from '@studio/constants/featureFlags/featureFlags';

// Capture the real flags before any mocking occurs
const originalFlags = { ...featureFlagsModule.featureFlags };

/**
 * Override specific feature flags for a test. Call in `beforeEach` or at the
 * top of an individual test. Automatically restored by the global
 * `vi.restoreAllMocks()` in afterEach.
 *
 * @example
 * beforeEach(() => {
 *   mockFeatureFlags({ toolCallingEnabled: true });
 * });
 */
export const mockFeatureFlags = (overrides: Partial<FeatureFlags>) => {
  vi.spyOn(featureFlagsModule, 'featureFlags', 'get').mockReturnValue({
    ...originalFlags,
    ...overrides,
  });
};
