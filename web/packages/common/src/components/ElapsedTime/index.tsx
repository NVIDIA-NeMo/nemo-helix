// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EM_DASH } from '@nemo/common/src/constants';
import { useLiveSeconds } from '@nemo/common/src/hooks/useLiveSeconds';
import { formatTimeInSeconds, utcToLocalDate } from '@nemo/common/src/utils/date';
import { Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface ElapsedTimeProps {
  start?: string;
  end?: string;
  running: boolean;
}

/** Live count from `start` while `running`, else `start` → `end`. Key by item in lists: the
 * live count locks onto the first start it sees. */
export const ElapsedTime: FC<ElapsedTimeProps> = ({ start, end, running }) => {
  const startDate = utcToLocalDate(start);
  const liveSeconds = useLiveSeconds({ startDate, enabled: running });

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
  const seconds = (endDate.getTime() - startDate.getTime()) / 1000;
  return <Text className="tabular-nums">{formatTimeInSeconds(seconds) || '0s'}</Text>;
};
