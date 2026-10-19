// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  canFineTuneModel,
  useModelCustomizationEligibility,
} from '@studio/hooks/useModelCustomizationEligibility';
import { renderHook } from '@testing-library/react';

const buildModel = (overrides: Partial<ModelEntity> = {}) =>
  ({ id: 'model-1', name: 'my-model', workspace: 'ws', ...overrides }) as ModelEntity;

describe('useModelCustomizationEligibility', () => {
  it('canFineTune=true when model has a fileset', () => {
    const { result } = renderHook(() =>
      useModelCustomizationEligibility(buildModel({ fileset: 'ws/my-fs' }))
    );
    expect(result.current.canFineTune).toBe(true);
    expect(result.current.canCustomize).toBe(true);
  });

  it('canFineTune=false when model has no fileset', () => {
    const { result } = renderHook(() => useModelCustomizationEligibility(buildModel()));
    expect(result.current.canFineTune).toBe(false);
    expect(result.current.canCustomize).toBe(false);
  });

  it('canFineTune=false when no model is given', () => {
    const { result } = renderHook(() => useModelCustomizationEligibility(undefined));
    expect(result.current.canFineTune).toBe(false);
  });

  it('never reports loading — eligibility is derived synchronously from the model', () => {
    const { result } = renderHook(() =>
      useModelCustomizationEligibility(buildModel({ fileset: 'ws/my-fs' }))
    );
    expect(result.current.isLoading).toBe(false);
  });

  describe('canFineTuneModel', () => {
    it('requires a fileset', () => {
      expect(canFineTuneModel(buildModel({ fileset: 'ws/my-fs' }))).toBe(true);
      expect(canFineTuneModel(buildModel())).toBe(false);
      expect(canFineTuneModel(null)).toBe(false);
      expect(canFineTuneModel(undefined)).toBe(false);
    });
  });
});
