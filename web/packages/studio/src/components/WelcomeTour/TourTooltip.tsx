// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from '@nvidia/foundations-react-core';
import { TourCard } from '@studio/components/WelcomeTour/TourCard';
import type { TooltipCoords } from '@studio/components/WelcomeTour/types';
import { computePosition } from '@studio/components/WelcomeTour/utils';
import { useTour } from 'modern-tour';
import { type FC, useLayoutEffect, useRef, useState } from 'react';

interface TourTooltipProps {
  onClose: () => void;
}

export const TourTooltip: FC<TourTooltipProps> = ({ onClose }) => {
  const { step, currentStep, totalSteps, targetRect, next, prev } = useTour();
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [coords, setCoords] = useState<TooltipCoords | null>(null);

  const isFirstStep = currentStep === 0;
  const isLastStep = currentStep === totalSteps - 1;

  useLayoutEffect(() => {
    if (!tooltipRef.current || !targetRect || !step) {
      setCoords(null);
      return;
    }
    const rect = tooltipRef.current.getBoundingClientRect();
    setCoords(computePosition(targetRect, rect, step.position ?? 'bottom'));
  }, [targetRect, step, currentStep]);

  if (!step || !targetRect) return null;

  return (
    <div
      ref={tooltipRef}
      className={`fixed z-[10000] w-[360px] max-w-[calc(100vw-32px)] ${coords ? '' : 'invisible'}`}
      style={coords ? { left: coords.left, top: coords.top } : undefined} // eslint-disable-line no-restricted-syntax
    >
      <TourCard
        title={typeof step.title === 'string' ? step.title : ''}
        body={step.content}
        stepLabel={`${currentStep + 1} of ${totalSteps}`}
        onClose={onClose}
        actions={
          <>
            {isFirstStep ? (
              <Button kind="tertiary" color="neutral" size="small" onClick={onClose}>
                Skip
              </Button>
            ) : (
              <Button kind="secondary" color="neutral" size="small" onClick={prev}>
                Back
              </Button>
            )}
            <Button kind="primary" color="brand" size="small" onClick={isLastStep ? onClose : next}>
              {isLastStep ? 'Done' : 'Next'}
            </Button>
          </>
        }
      />
    </div>
  );
};
