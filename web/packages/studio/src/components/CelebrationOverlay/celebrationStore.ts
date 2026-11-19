// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CELEBRATION_DURATION_MS } from '@studio/components/CelebrationOverlay/constants';

type Listener = () => void;

const FAILSAFE_MS = CELEBRATION_DURATION_MS + 500;

let activeCelebration: number | null = null;
let nextCelebration = 1;
const listeners = new Set<Listener>();

const emit = () => {
  for (const listener of listeners) {
    listener();
  }
};

export const dismissCelebration = () => {
  if (activeCelebration === null) {
    return;
  }
  activeCelebration = null;
  emit();
};

/**
 * Lives outside React so a launch can trigger it from a component that is about to unmount on
 * navigation; `CelebrationHost` at the app root renders it.
 */
export const celebrate = () => {
  const celebration = nextCelebration;
  activeCelebration = celebration;
  nextCelebration += 1;
  emit();
  // Clears the overlay even if its own timer never ran because the overlay failed to render.
  window.setTimeout(() => {
    if (activeCelebration === celebration) {
      dismissCelebration();
    }
  }, FAILSAFE_MS);
};

export const subscribeToCelebration = (listener: Listener) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};

export const getCelebration = () => activeCelebration;
