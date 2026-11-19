// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)';

/** Tracks the user's "reduce motion" preference. Safe where matchMedia is unavailable (jsdom). */
export const usePrefersReducedMotion = (): boolean => {
  const [reduced, setReduced] = useState(
    () => typeof window.matchMedia === 'function' && window.matchMedia(REDUCED_MOTION_QUERY).matches
  );

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const query = window.matchMedia(REDUCED_MOTION_QUERY);
    const handleChange = (event: MediaQueryListEvent) => setReduced(event.matches);
    query.addEventListener('change', handleChange);
    return () => query.removeEventListener('change', handleChange);
  }, []);

  return reduced;
};
