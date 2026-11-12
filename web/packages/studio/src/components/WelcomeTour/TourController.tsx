// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelpCircle } from '@nemo/icons/icons';
import { Button, Tooltip } from '@nvidia/foundations-react-core';
import { TourTooltip } from '@studio/components/WelcomeTour/TourTooltip';
import { studioPreferences } from '@studio/providers/preferences/definitions';
import { usePreference } from '@studio/providers/preferences/usePreference';
import { useTour } from 'modern-tour';
import { type FC, useCallback, useEffect } from 'react';

export const TourController: FC = () => {
  const [tourSeen, setTourSeen, , tourSeenKnown] = usePreference(studioPreferences.tourSeen);
  const { start, stop, isOpen } = useTour();

  const startTour = useCallback(() => {
    start(0);
  }, [start]);

  const handleClose = useCallback(() => {
    stop();
    setTourSeen(true);
  }, [stop, setTourSeen]);

  // Auto-start on first visit
  useEffect(() => {
    if (tourSeen || !tourSeenKnown) return;
    const timer = setTimeout(() => {
      start(0);
    }, 800);
    return () => clearTimeout(timer);
  }, [tourSeen, tourSeenKnown, start]);

  return (
    <>
      <Tooltip slotContent="Take a tour" side="bottom">
        <Button
          aria-label="Take a tour"
          color="neutral"
          kind="tertiary"
          size="medium"
          onClick={startTour}
        >
          <HelpCircle />
        </Button>
      </Tooltip>

      {isOpen && <TourTooltip onClose={handleClose} />}
    </>
  );
};
