// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EM_DASH } from '@nemo/common/src/constants';
import { useLiveSeconds } from '@nemo/common/src/hooks/useLiveSeconds';
import { formatTimeInSeconds, utcToLocalDate } from '@nemo/common/src/utils/date';
import { Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface ElapsedTimeProps {
  /** When the clock started, e.g. a job's `created_at`. Timestamps without a zone are UTC. */
  start?: string;
  /** When the clock stopped, e.g. a finished job's `updated_at`. Ignored while `running`. */
  end?: string;
  /** Count up live from `start` instead of showing the fixed `start` → `end` time. */
  running: boolean;
}

/**
 * How long something has run: a live count from `start` while `running`, otherwise the fixed time
 * from `start` to `end`. In a list, key it by item: `useLiveSeconds` locks onto the first start it
 * sees, so a reused instance would keep counting from another item's start.
 */
export const ElapsedTime: FC<ElapsedTimeProps> = ({ start, end, running }) => {
  const startDate = utcToLocalDate(start);
  const liveSeconds = useLiveSeconds({ startDate, enabled: running });

  if (!startDate) return <Text>{EM_DASH}</Text>;

  if (running) {
    return <Text className="tabular-nums">{formatTimeInSeconds(liveSeconds) || '0s'}</Text>;
  }

  const endDate = utcToLocalDate(end);
  if (!endDate) return <Text>{EM_DASH}</Text>;
  // Whole seconds, like the live count, so stopped and running rows read alike.
  const seconds = (endDate.getTime() - startDate.getTime()) / 1000;
  return <Text className="tabular-nums">{formatTimeInSeconds(seconds) || '0s'}</Text>;
};
