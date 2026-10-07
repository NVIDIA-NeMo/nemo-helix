// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Tooltip } from '@nvidia/foundations-react-core';
import { TourTooltip } from '@studio/components/WelcomeTour/TourTooltip';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { TOUR_SEEN_KEY } from '@studio/util/localStorage';
import { CircleHelp } from 'lucide-react';
import { useTour } from 'modern-tour';
import { FC, useCallback, useEffect } from 'react';

export const TourController: FC = () => {
  const [tourSeen, setTourSeen] = useLocalStorage<boolean>(TOUR_SEEN_KEY);
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
    if (tourSeen) return;
    const timer = setTimeout(() => {
      start(0);
    }, 800);
    return () => clearTimeout(timer);
  }, [tourSeen, start]);

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
          <CircleHelp />
        </Button>
      </Tooltip>

      {isOpen && <TourTooltip onClose={handleClose} />}
    </>
  );
};
