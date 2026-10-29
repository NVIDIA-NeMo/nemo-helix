// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SharedModelEntry } from '@studio/routes/ModelCompareRoute/types';
import { useCallback, useRef, useState } from 'react';

export const MAX_COMPARE_MODELS = 4;

/** The models open side by side in the compare view, with an empty slot for each unpicked one. */
export const useCompareModels = (initial: Omit<SharedModelEntry, 'id'>[]) => {
  const [models, setModels] = useState<SharedModelEntry[]>(() =>
    initial.map((entry, id) => ({ ...entry, id }))
  );
  const nextIdRef = useRef(initial.length);

  const addModel = useCallback(() => {
    setModels((prev) =>
      prev.length >= MAX_COMPARE_MODELS
        ? prev
        : [...prev, { id: nextIdRef.current++, modelURN: null }]
    );
  }, []);

  const removeModel = useCallback((id: number) => {
    setModels((prev) => prev.filter((m) => m.id !== id));
  }, []);

  const setModel = useCallback((id: number, modelURN: string | null, adapter?: string | null) => {
    setModels((prev) =>
      prev.map((m) => (m.id === id ? { ...m, modelURN, adapter: adapter ?? null } : m))
    );
  }, []);

  return {
    models,
    setModels,
    addModel: models.length < MAX_COMPARE_MODELS ? addModel : undefined,
    removeModel,
    setModel,
  };
};
