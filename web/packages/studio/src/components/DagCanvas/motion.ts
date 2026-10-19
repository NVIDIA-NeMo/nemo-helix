// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const GRAPH_MOTION_DURATION_MS = 500;

export const getGraphMotionDuration = (): number =>
  typeof window !== 'undefined' &&
  typeof window.matchMedia === 'function' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ? 0
    : GRAPH_MOTION_DURATION_MS;
