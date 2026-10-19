// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import {
  type Adapter,
  FinetuningType,
  ModelDeploymentStatus,
  type ModelEntity,
} from '@nemo/sdk/generated/platform/schema';
import { useModelChatAvailability } from '@studio/hooks/useModelChatAvailability';
import { useModelDeploymentStatus } from '@studio/hooks/useModelDeploymentStatus';
import { useModelIsServed } from '@studio/hooks/useModelIsServed';
import { renderHook } from '@testing-library/react';

vi.mock('@nemo/sdk/generated/platform/models');
vi.mock('@studio/hooks/useModelDeploymentStatus');
vi.mock('@studio/hooks/useModelIsServed');

const mockGetModel = vi.mocked(useModelsGetModel);
const mockDeploymentStatus = vi.mocked(useModelDeploymentStatus);
const mockIsServed = vi.mocked(useModelIsServed);

/** Created well outside the five-minute grace window, so status is the real answer. */
const model = (overrides: Partial<ModelEntity> = {}): ModelEntity => ({
  id: 'm-1',
  name: 'flying-tomato-viper',
  workspace: 'default',
  created_at: '2020-01-01T00:00:00Z',
  updated_at: '2020-01-01T00:00:00Z',
  ...overrides,
});

describe('useModelChatAvailability', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetModel.mockReturnValue({ data: undefined, isLoading: false } as ReturnType<
      typeof useModelsGetModel
    >);
    mockDeploymentStatus.mockReturnValue({ status: ModelDeploymentStatus.READY, isLoading: false });
    mockIsServed.mockReturnValue({ isServed: true, isLoading: false, isError: false });
  });

  // The reported bug. A full-weight run sets `base_model` like any fine-tune, but it
  // owns its deployment; resolving against the base reported it as unavailable.
  it('resolves a full-weight fine-tune against itself, not its base', () => {
    const entity = model({
      base_model: 'default/qwen3-0.6b',
      finetuning_type: FinetuningType.all_weights,
    });

    const { result } = renderHook(() => useModelChatAvailability(entity));

    // No base lookup at all: the query is disabled, so nothing 404s on the way.
    expect(mockGetModel).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      undefined,
      expect.objectContaining({ query: expect.objectContaining({ enabled: false }) })
    );
    expect(mockDeploymentStatus).toHaveBeenCalledWith(entity);
    expect(result.current.modelChatStatus).toBe('enabled');
    expect(result.current.isChatAvailable).toBe(true);
  });

  // An adapter really is served by its base, so that redirect must survive the fix.
  it('resolves a LoRA adapter against its base model', () => {
    const base = model({ name: 'qwen3-0.6b' });
    mockGetModel.mockReturnValue({ data: base, isLoading: false } as ReturnType<
      typeof useModelsGetModel
    >);

    renderHook(() =>
      useModelChatAvailability(
        model({ base_model: 'default/qwen3-0.6b', finetuning_type: FinetuningType.lora })
      )
    );

    // Workspace and name taken apart: passing the whole ref as the name 404s.
    expect(mockGetModel).toHaveBeenCalledWith('default', 'qwen3-0.6b', undefined, {
      query: { enabled: true, retry: false },
    });
    expect(mockDeploymentStatus).toHaveBeenCalledWith(base);
  });

  // A model whose deployment really is gone must still report unavailable — the fix
  // changes which entity is asked, not whether the answer is believed.
  it('still reports disabled when the model has no deployment', () => {
    mockDeploymentStatus.mockReturnValue({ status: null, isLoading: false });
    mockIsServed.mockReturnValue({ isServed: false, isLoading: false, isError: false });

    const { result } = renderHook(() =>
      useModelChatAvailability(
        model({ base_model: 'default/qwen3-0.6b', finetuning_type: FinetuningType.all_weights })
      )
    );

    expect(result.current.modelChatStatus).toBe('disabled');
  });
  // An adapter row's `model` is already the base, so the deployment question is about
  // the base — but "is it served" is about the adapter. A READY base deployment that
  // has not loaded the adapter serves the base and nothing else.
  describe('adapter rows', () => {
    const base = model({ name: 'qwen3-0.6b' });
    const adapter: Adapter = { name: 'support-v1', workspace: 'default' } as Adapter;

    it('asks whether the adapter is served, not its base', () => {
      renderHook(() => useModelChatAvailability(base, { adapter }));

      expect(mockIsServed).toHaveBeenCalledWith(base, adapter);
      // The base is the parent here, so it is checked directly — no second lookup.
      expect(mockGetModel).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        undefined,
        expect.objectContaining({ query: expect.objectContaining({ enabled: false }) })
      );
    });

    it('reports an adapter its provider has not loaded as unserved', () => {
      mockIsServed.mockReturnValue({ isServed: false, isLoading: false, isError: false });

      const { result } = renderHook(() => useModelChatAvailability(base, { adapter }));

      // Base deployment is READY (the default), so only the adapter check catches this.
      expect(result.current.isAdapterUnserved).toBe(true);
      expect(result.current.isChatAvailable).toBe(false);
    });

    it('enables chat once the adapter appears in served_models', () => {
      const { result } = renderHook(() => useModelChatAvailability(base, { adapter }));

      expect(result.current.isAdapterUnserved).toBe(false);
      expect(result.current.modelChatStatus).toBe('enabled');
    });

    // `isAdapterUnserved` drives a distinct empty state, so it must not fire while the
    // provider lookup is still in flight.
    it('is not unserved while the provider lookup is still loading', () => {
      mockIsServed.mockReturnValue({ isServed: false, isLoading: true, isError: false });

      const { result } = renderHook(() => useModelChatAvailability(base, { adapter }));

      expect(result.current.isAdapterUnserved).toBe(false);
    });

    // Provider queries do not retry, so a single 5xx lands as isServed:false. Claiming
    // the base deployment has not loaded the adapter would then be a confident guess.
    it('does not report unserved when the provider lookup failed', () => {
      mockIsServed.mockReturnValue({ isServed: false, isLoading: false, isError: true });

      const { result } = renderHook(() => useModelChatAvailability(base, { adapter }));

      expect(result.current.isAdapterUnserved).toBe(false);
    });

    it('never reports unserved for a plain model', () => {
      mockIsServed.mockReturnValue({ isServed: false, isLoading: false, isError: false });

      const { result } = renderHook(() => useModelChatAvailability(model()));

      expect(result.current.isAdapterUnserved).toBe(false);
    });
  });
});
