// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ColorMode } from '@xyflow/react';
import { useEffect, useState } from 'react';

const readColorMode = (): ColorMode =>
  typeof document !== 'undefined' && document.documentElement.classList.contains('nv-dark')
    ? 'dark'
    : 'light';

/**
 * Tracks the active Studio theme by watching the `nv-dark` class on `<html>` (the
 * same signal that drives NVIDIA Foundations design tokens) and returns the matching
 * React Flow {@link ColorMode}.
 *
 * React Flow scopes its rendering surface with a `light`/`dark` class derived from its
 * `colorMode` prop, which otherwise defaults to `light` and overrides the app's dark
 * tokens inside the canvas — leaving nodes white-on-dark. Feeding this value back into
 * `colorMode` keeps the canvas in sync with the rest of the UI.
 */
export const useNvColorMode = (): ColorMode => {
  const [colorMode, setColorMode] = useState<ColorMode>(readColorMode);

  useEffect(() => {
    const update = () => setColorMode(readColorMode());
    update();
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });
    return () => observer.disconnect();
  }, []);

  return colorMode;
};
