// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import {
  dismissCelebration,
  getCelebration,
  subscribeToCelebration,
} from '@studio/components/CelebrationOverlay/celebrationStore';
import {
  CELEBRATION_DURATION_MS,
  CELEBRATION_MESSAGE,
  REDUCED_MOTION_DURATION_MS,
} from '@studio/components/CelebrationOverlay/constants';
import { Fireworks } from '@studio/components/CelebrationOverlay/Fireworks';
import {
  Component,
  type FC,
  type ReactNode,
  useEffect,
  useState,
  useSyncExternalStore,
} from 'react';
import { createPortal } from 'react-dom';

const prefersReducedMotion = () =>
  typeof window.matchMedia === 'function' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

interface ErrorBoundaryState {
  readonly hasError: boolean;
}

class CelebrationErrorBoundary extends Component<
  { readonly children: ReactNode },
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  override render() {
    return this.state.hasError ? null : this.props.children;
  }
}

export const CelebrationOverlay: FC<{ onDismiss: () => void }> = ({ onDismiss }) => {
  const [reducedMotion] = useState(prefersReducedMotion);

  useEffect(() => {
    const timer = window.setTimeout(
      onDismiss,
      reducedMotion ? REDUCED_MOTION_DURATION_MS : CELEBRATION_DURATION_MS
    );
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onDismiss();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [onDismiss, reducedMotion]);

  return createPortal(
    <div
      data-testid="celebration-overlay"
      onClick={onDismiss}
      className="fixed inset-0 z-[2147483647] flex cursor-pointer items-center justify-center bg-black/60"
    >
      {reducedMotion ? null : (
        <CelebrationErrorBoundary>
          <Fireworks />
        </CelebrationErrorBoundary>
      )}
      <Text
        kind="display/xl"
        role="status"
        aria-live="polite"
        className="relative px-density-2xl text-center font-bold text-brand"
      >
        {CELEBRATION_MESSAGE}
      </Text>
    </div>,
    document.body
  );
};

/** Mounted once at the app root so the overlay outlives the page that launched it. */
export const CelebrationHost: FC = () => {
  const celebration = useSyncExternalStore(subscribeToCelebration, getCelebration);
  if (celebration === null) {
    return null;
  }
  return (
    <CelebrationErrorBoundary>
      <CelebrationOverlay key={celebration} onDismiss={dismissCelebration} />
    </CelebrationErrorBoundary>
  );
};
