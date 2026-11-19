// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type DateStringISO,
  getDifferenceBetween,
  formatAbsoluteTimestamp,
  getTimeRelativeToNow,
  parseISOWithUTCFallback,
} from '@nemo/common/src/components/RelativeTime/util';
import { Text, Tooltip } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { add } from 'date-fns';
import { useEffect, useRef, useState } from 'react';

type RelativeTimeProps = {
  datetime: DateStringISO;
  align?: 'center' | 'left' | 'right';
  underline?: boolean;
  abbreviated?: boolean;
  /**
   * When true (default), the timestamp is focusable (`tabIndex={0}`) so keyboard users can open
   * the absolute-time tooltip on focus. Set to false in modal/side-panel content where focus
   * traps would move focus here on open and incorrectly show the tooltip.
   */
  focusableForTooltip?: boolean;
};

/**
 * returns the ms until the next whole number interval of time
 * returns `null` if more than a day has passed/until */
export const nextIntervalFrom = (start: DateStringISO): null | number => {
  const date = parseISOWithUTCFallback(start);
  const now = new Date(Date.now());

  const difference = getDifferenceBetween(now, date);

  const isPast = difference.milliseconds > 0;

  if (Math.abs(difference.days) > 0) {
    // if more than a day away (past or future), we don't need a live ticker
    return null;
  }

  if (Math.abs(difference.hours) > 0) {
    // For past: update at the next hour mark going forward from the date
    // For future: update when it crosses into the next smaller hour count
    const hoursToAdd = isPast ? difference.hours + 1 : difference.hours;
    const target = add(date, { hours: hoursToAdd });
    return target.getTime() - now.getTime();
  }

  // Same logic for minutes
  const minutesToAdd = isPast ? difference.minutes + 1 : difference.minutes;
  const target = add(date, { minutes: minutesToAdd });
  return target.getTime() - now.getTime();
};

export const useRelativeTimeSince = (datetime: DateStringISO, abbreviated = false) => {
  const [timeSince, setTimeSince] = useState(getTimeRelativeToNow(datetime, abbreviated));
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    setTimeSince(getTimeRelativeToNow(datetime, abbreviated));

    const timeUntilNext = nextIntervalFrom(datetime);
    let timer: ReturnType<typeof setTimeout>;
    if (timeUntilNext === null) {
      return;
    }

    /** recursively update the timeout timer and update the time-since string */
    const recursiveTimeSinceUpdater = () => {
      if (!isMountedRef.current) return;
      setTimeSince(getTimeRelativeToNow(datetime, abbreviated));
      const timeUntilNext = nextIntervalFrom(datetime);
      if (timeUntilNext !== null) {
        timer = setTimeout(recursiveTimeSinceUpdater, timeUntilNext);
      }
    };

    timer = setTimeout(recursiveTimeSinceUpdater, timeUntilNext);

    return () => {
      isMountedRef.current = false;
      clearTimeout(timer);
    };
  }, [datetime, abbreviated]);

  return timeSince;
};

export const RelativeTime = ({
  datetime,
  align,
  underline = true,
  abbreviated = false,
  focusableForTooltip = true,
}: RelativeTimeProps) => {
  const distance = useRelativeTimeSince(datetime, abbreviated);

  return (
    <Tooltip slotContent={formatAbsoluteTimestamp(datetime)}>
      <Text
        asChild
        {...(focusableForTooltip ? { tabIndex: 0 as const } : {})}
        className={cn(
          'text-inherit lining-nums tabular-nums whitespace-nowrap',
          align === 'center' ? 'text-center' : align === 'right' ? 'text-right' : 'text-left',
          underline && 'underline decoration-dashed decoration-(--border-color-base)'
        )}
      >
        <time dateTime={datetime}>{distance}</time>
      </Text>
    </Tooltip>
  );
};
