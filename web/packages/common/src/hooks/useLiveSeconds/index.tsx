// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useEffect } from 'react';

import { getElapsedSeconds } from '../../utils/date';

interface UseLiveSecondsProps {
  startDate?: Date;
  enabled?: boolean;
}

/**
 * Use this hook to get the live seconds since the start time.
 * If startDate is not provided, the hook returns undefined (so you don't render a non-relative time).
 * Once startDate is provided, the hook will start returning the difference in seconds between the startDate and the current date.
 * This behavior is useful for fetching job details async and then rendering the relative time.
 */
export const useLiveSeconds = ({ startDate, enabled = true }: UseLiveSecondsProps) => {
  const startMs = enabled ? startDate?.getTime() : undefined;
  const [, setTick] = useState(0);

  useEffect(() => {
    if (startMs === undefined) {
      return;
    }

    const interval = setInterval(() => setTick((tick) => tick + 1), 1000);
    return () => clearInterval(interval);
  }, [startMs]);

  return startMs === undefined ? undefined : getElapsedSeconds(startMs);
};
