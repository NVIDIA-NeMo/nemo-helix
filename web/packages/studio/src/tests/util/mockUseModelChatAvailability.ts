// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelChatStatus } from '@nemo/common/src/utils/models';
import * as useModelChatAvailabilityModule from '@studio/hooks/useModelChatAvailability';

interface MockUseModelChatAvailabilityOptions {
  modelChatStatus?: ModelChatStatus;
  isChatAvailable?: boolean;
  isLoading?: boolean;
  isAdapterUnserved?: boolean;
}

const defaults: Required<MockUseModelChatAvailabilityOptions> = {
  modelChatStatus: 'enabled',
  isChatAvailable: true,
  isLoading: false,
  isAdapterUnserved: false,
};

export const mockUseModelChatAvailability = (overrides?: MockUseModelChatAvailabilityOptions) => {
  vi.spyOn(useModelChatAvailabilityModule, 'useModelChatAvailability').mockReturnValue({
    ...defaults,
    ...overrides,
  });
};

// Auto-apply default mock before each test so any test importing this helper
// gets a stable 'enabled' status without each test having to opt in. Overrides
// from per-test mockUseModelChatAvailability(...) calls take precedence.
beforeEach(() => {
  mockUseModelChatAvailability();
});
