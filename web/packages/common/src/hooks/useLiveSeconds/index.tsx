// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getElapsedSeconds } from '@nemo/common/src/utils/date';
import { useState, useEffect } from 'react';

interface UseLiveSecondsProps {
  startDate?: Date;
}

/**
 * Use this hook to get the live seconds since the start time.
 * If startDate is not provided, the hook returns undefined (so you don't render a non-relative time).
 * Once startDate is provided, the hook will start returning the difference in seconds between the startDate and the current date.
 * This behavior is useful for fetching job details async and then rendering the relative time.
 */
export const useLiveSeconds = ({ startDate }: UseLiveSecondsProps) => {
  const startMs = startDate?.getTime();
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
