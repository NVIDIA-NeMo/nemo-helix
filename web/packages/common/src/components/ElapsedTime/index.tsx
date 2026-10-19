// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EM_DASH } from '@nemo/common/src/constants';
import { useLiveSeconds } from '@nemo/common/src/hooks/useLiveSeconds';
import {
  formatDurationMs,
  formatTimeInSeconds,
  getElapsedSeconds,
  utcToLocalDate,
} from '@nemo/common/src/utils/date';
import { Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface ElapsedTimeProps {
  start?: string;
  end?: string;
  /** Recorded duration; when stopped, shown instead of `start` → `end`. */
  durationMs?: number;
  running: boolean;
}

/** Live count from `start` while `running`, else `durationMs`, else `start` → `end`. */
export const ElapsedTime: FC<ElapsedTimeProps> = ({ start, end, durationMs, running }) => {
  const startDate = utcToLocalDate(start);
  const liveSeconds = useLiveSeconds({ startDate, enabled: running });

  if (!running && durationMs != null) {
    return (
      <Text className="tabular-nums">
        {formatDurationMs(durationMs, { hideMsAboveMinute: true })}
      </Text>
    );
  }

  if (!startDate) {
    return <Text>{EM_DASH}</Text>;
  }

  if (running) {
    return <Text className="tabular-nums">{formatTimeInSeconds(liveSeconds) || '0s'}</Text>;
  }

  const endDate = utcToLocalDate(end);
  if (!endDate) {
    return <Text>{EM_DASH}</Text>;
  }
  const seconds = getElapsedSeconds(startDate.getTime(), endDate.getTime());
  return <Text className="tabular-nums">{formatTimeInSeconds(seconds) || '0s'}</Text>;
};
