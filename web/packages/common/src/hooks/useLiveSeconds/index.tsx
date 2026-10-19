// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useEffect } from 'react';

interface UseLiveSecondsProps {
  startDate?: Date;
  enabled?: boolean;
}

const secondsSince = (start: Date): number => Math.floor((Date.now() - start.getTime()) / 1000);

/**
 * Use this hook to get the live seconds since the start time.
 * If startDate is not provided, the hook returns undefined (so you don't render a non-relative time).
 * Once startDate is provided, the hook will start returning the difference in seconds between the startDate and the current date.
 * This behavior is useful for fetching job details async and then rendering the relative time.
 */
export const useLiveSeconds = ({ startDate, enabled = true }: UseLiveSecondsProps) => {
  const [liveSeconds, setLiveSeconds] = useState(0);
  const [lockedStartDate, setLockedStartDate] = useState<Date | undefined>(undefined);

  // Only start counting when startDate is provided
  useEffect(() => {
    if (startDate === undefined || !enabled) {
      return;
    }

    if (lockedStartDate === undefined) {
      // Lock the startDate value to prevent desync when startDate is provided asynchronously
      setLockedStartDate(startDate);
      setLiveSeconds(secondsSince(startDate));
    }
  }, [startDate, lockedStartDate, enabled]);

  // Separate effect for the interval to avoid creating multiple timers
  useEffect(() => {
    if (lockedStartDate === undefined || !enabled) {
      return;
    }

    setLiveSeconds(secondsSince(lockedStartDate));
    const interval = setInterval(() => setLiveSeconds(secondsSince(lockedStartDate)), 1000);
    return () => clearInterval(interval);
  }, [lockedStartDate, enabled]);

  if (lockedStartDate !== undefined) {
    return liveSeconds;
  }
  // The start locks only after the first render; read the clock until then.
  return startDate !== undefined && enabled ? secondsSince(startDate) : undefined;
};
