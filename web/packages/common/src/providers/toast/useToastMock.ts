// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { vi } from 'vitest';

import { ToastObject } from './types';

/**
 * Builds a ToastObject with every method stubbed as a vi.fn(), so tests only
 * need to override the methods they actually assert on.
 *
 * vi.mock is hoisted and statically analyzed per test file, so each test still
 * calls vi.mock itself (auto-mocked, no factory needed). Assign a fresh mock
 * to the mocked useToast in beforeEach and keep a handle to assert against:
 *
 *   import { useToast } from '@nemo/common/src/providers/toast/useToast';
 *   import { createToastMock } from '@nemo/common/src/providers/toast/useToastMock';
 *
 *   vi.mock('@nemo/common/src/providers/toast/useToast');
 *
 *   let toast = createToastMock();
 *   beforeEach(() => {
 *     toast = createToastMock();
 *     vi.mocked(useToast).mockReturnValue(toast);
 *   });
 *
 *   expect(toast.success).toHaveBeenCalled();
 */
export const createToastMock = (overrides: Partial<ToastObject> = {}): ToastObject => ({
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  working: vi.fn(),
  workingWithId: vi.fn(() => 'mock-toast-id'),
  neutral: vi.fn(),
  dismissToast: vi.fn(),
  ...overrides,
});
