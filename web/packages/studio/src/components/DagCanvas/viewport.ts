// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Viewport } from '@xyflow/react';

export const MIN_GRAPH_ZOOM = 0.001;

export const readViewport = (key: string): Viewport | null => {
  try {
    const viewport = JSON.parse(sessionStorage.getItem(key) ?? '') as Partial<Viewport>;
    return typeof viewport.x === 'number' &&
      typeof viewport.y === 'number' &&
      typeof viewport.zoom === 'number'
      ? (viewport as Viewport)
      : null;
  } catch {
    return null;
  }
};
