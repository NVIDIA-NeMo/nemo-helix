// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DateFns from 'date-fns';

export type DateStringISO = string;

/**
 * Normalizes an ISO datetime string to ensure timezone-naive strings are treated as UTC.
 * If the datetime string doesn't have timezone information (Z or ±HH:MM offset),
 * appends 'Z' to indicate UTC timezone.
 *
 * @param datetime - ISO 8601 datetime string
 * @returns Normalized datetime string with explicit timezone
 */
const normalizeToUTC = (datetime: string): string => {
  const hasTimezone = /Z|[+-]\d{2}:\d{2}$/.test(datetime);

  if (hasTimezone) {
    return datetime;
  }

  // If no timezone, append 'Z' to treat as UTC
  return datetime + 'Z';
};

/**
 * Parses an ISO datetime string with UTC fallback for timezone-naive strings.
 * If the datetime string doesn't have timezone information, it's treated as UTC.
 *
 * @param datetime - ISO 8601 datetime string
 * @returns Parsed Date object
 */
export const parseISOWithUTCFallback = (datetime: string): Date =>
  DateFns.parseISO(normalizeToUTC(datetime));

export type UnitsOfTime =
  | 'years'
  | 'months'
  | 'weeks'
  | 'days'
  | 'hours'
  | 'minutes'
  | 'seconds'
  | 'milliseconds';

/** It is desired to wrap around the compareDesc to allow
 * the acceptance of ISO formatted datetime strings when using compareDesc
 * @param stringOne - first datestring to compare
 * @param stringTwo - second datestring to compare
 *
 * @returns Compare the two dates and return -1 if the first date is after the second,
 * 1 if the first date is before the second or 0 if dates are equal.
 */
export const compareDescStringISO = (stringOne: DateStringISO, stringTwo: DateStringISO) =>
  DateFns.compareDesc(parseISOWithUTCFallback(stringOne), parseISOWithUTCFallback(stringTwo));

export const getDifferenceBetween = (a: Date, b: Date): Record<UnitsOfTime, number> => {
  const milliseconds = DateFns.differenceInMilliseconds(a, b);
  const seconds = DateFns.differenceInSeconds(a, b);
  const minutes = DateFns.differenceInMinutes(a, b);
  const hours = DateFns.differenceInHours(a, b);
  const days = DateFns.differenceInDays(a, b);
  const weeks = DateFns.differenceInWeeks(a, b);
  const months = DateFns.differenceInMonths(a, b);
  const years = DateFns.differenceInYears(a, b);

  return {
    milliseconds,
    seconds,
    minutes,
    hours,
    days,
    weeks,
    months,
    years,
  };
};
/** It is desired to wrap around the isBefore func to allow
 * the use of ISO formatted string and compare to now
 * @param stringISO - first datestring to use
 *
 * @returns Boolean - Is the first date before the second one
 */
export const isBeforeNowStringISO = (stringISO: DateStringISO) =>
  DateFns.compareDesc(parseISOWithUTCFallback(stringISO), new Date());

/** Get a shorthand version of formatDistance
 * If the distance is more than a month we will show the international date.
 * @param start:String - ISO timestamp
 * @param end:String - ISO timestamp to compare against
 * @param addSuffix:Boolean - bool to add suffix or not
 *
 * @return String - shorthand version of what formatDistanceStrict would return
 */
type TimeDistance = {
  start: DateStringISO;
  end: DateStringISO;
  addSuffix?: boolean;
};
export const getShorthandDistance = ({ start, end, addSuffix }: TimeDistance) => {
  if (!start || !end) {
    return '';
  }
  const startDate = parseISOWithUTCFallback(start);
  const endDate = parseISOWithUTCFallback(end);

  if (DateFns.differenceInMonths(endDate, startDate) > 0) {
    return DateFns.intlFormat(startDate, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  }

  if (addSuffix) {
    const longhand = DateFns.formatDistanceStrict(startDate, endDate, {
      addSuffix: true,
    });
    if (DateFns.compareAsc(startDate, endDate)) {
      const [amount, unit, suffix] = longhand.split(' ');
      return `${amount}${getUnitShorthand(unit)} ${suffix}`;
    } else {
      const [prefix, amount, unit] = longhand.split(' ');
      return `${prefix} ${amount}${getUnitShorthand(unit)}`;
    }
  }
  const [amount, unit] = DateFns.formatDistanceStrict(startDate, endDate).split(' ');
  return `${amount}${getUnitShorthand(unit)}`;
};

type TimeDistanceToNow = {
  date: TimeDistance['start'];
  addSuffix?: TimeDistance['addSuffix'];
};
export const getShorthandDistanceToNow = ({ date, addSuffix }: TimeDistanceToNow) =>
  getShorthandDistance({
    start: date,
    end: new Date().toISOString(),
    addSuffix,
  });

/** Crafting shorthand for twitter like 10s, 20m, 1hr
 * @param unit - String longhand version of unit
 *
 * @returns String - shorthand version of unit
 */
export const getUnitShorthand = (unit: string) => {
  switch (unit) {
    case 'second':
    case 'seconds':
      return 's';
    case 'minute':
      return ' min';
    case 'minutes':
      return ' mins';
    case 'hour':
    case 'hours':
      return 'h';
    case 'day':
    case 'days':
    default:
      return 'd';
    // If we've gotten to months they should be seeing date shorthand instead.
  }
};

/**
 * I need the start date of the next month
 * @returns String - The first of the next month in ISO format
 */
export const getStartOfNextMonth = (): DateStringISO =>
  DateFns.formatISO(DateFns.startOfMonth(DateFns.addMonths(new Date(), 1)));

/**
 * Formats a timestamp as an absolute datetime according to RFC spec.
 * - Current year: "MMM D at h:mm A z" (e.g., "Oct 27 at 07:09 PM EST")
 * - Prior years: "MMM D, YYYY at h:mm A z" (e.g., "Oct 27, 2024 at 07:09 PM EST")
 *
 * @param time DateString ISO 8601
 * @returns Formatted absolute timestamp string
 */
export const formatAbsoluteTimestamp = (time: DateStringISO): string => {
  const date = parseISOWithUTCFallback(time);
  const now = new Date();
  const isCurrentYear = date.getFullYear() === now.getFullYear();

  // Format: "MMM D at h:mm A z"
  const options: Intl.DateTimeFormatOptions = {
    month: 'short',
    day: 'numeric',
    ...(isCurrentYear ? {} : { year: 'numeric' }),
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
    timeZoneName: 'short',
  };

  const formatter = new Intl.DateTimeFormat('en-US', options);
  const formatted = formatter.format(date);

  // Intl.DateTimeFormat produces "Oct 27, 2024, 7:09 PM EST"
  // We need to transform it to "Oct 27 at 7:09 PM EST" or "Oct 27, 2024 at 7:09 PM EST"
  if (isCurrentYear) {
    // Replace first comma with " at"
    return formatted.replace(',', ' at');
  } else {
    // Replace second comma with " at" (keep first comma for year)
    return formatted.replace(/,([^,]*)$/, ' at$1');
  }
};

/**
 * Formats a timestamp as a relative time string.
 * Algorithm adapted from GitHub's relative-time-element:
 * https://github.com/github/relative-time-element/blob/main/src/duration.ts
 *
 * @param time DateString ISO 8601
 * @param abbreviated Whether to use abbreviated format (e.g., "2 h ago" vs "2 hours ago")
 * @returns A string describing the time relative to now.
 * (e.g. "just now", "2 minutes ago", "yesterday", "last month")
 */
export const getTimeRelativeToNow = (time?: DateStringISO, abbreviated = false) => {
  if (!time) {
    return '';
  }

  const date = parseISOWithUTCFallback(time);

  // check that `time` is a valid DateStringISO
  if (Number.isNaN(date.getTime())) {
    return '';
  }

  const now = new Date();
  const deltaMs = date.getTime() - now.getTime();
  const isPast = deltaMs < 0;
  const absMs = Math.abs(deltaMs);

  // Calculate raw differences
  const seconds = Math.floor(absMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  // Round to single unit (following GitHub's roundToSingleUnit logic)

  // < 55 seconds → "just now"
  if (seconds < 55) {
    return abbreviated ? 'now' : 'just now';
  }

  // >= 55 seconds AND < 55 minutes → minutes
  if (minutes < 55) {
    const count = minutes || 1;
    if (count === 1) {
      return abbreviated
        ? isPast
          ? '1 min ago'
          : 'in 1 min'
        : isPast
          ? 'a minute ago'
          : 'in a minute';
    }
    return abbreviated
      ? isPast
        ? `${count} min ago`
        : `in ${count} min`
      : isPast
        ? `${count} minutes ago`
        : `in ${count} minutes`;
  }

  // >= 55 minutes AND < 21 hours (or 12 hours if days exist) → hours
  const hourThreshold = days > 0 ? 12 : 21;
  if (hours < hourThreshold) {
    const count = Math.max(hours, 1);
    if (count === 1) {
      return abbreviated ? (isPast ? '1 h ago' : 'in 1 h') : isPast ? 'an hour ago' : 'in an hour';
    }
    return abbreviated
      ? isPast
        ? `${count} h ago`
        : `in ${count} h`
      : isPast
        ? `${count} hours ago`
        : `in ${count} hours`;
  }

  // >= 21 hours (or 12 w/days) AND < 6 days → days
  if (days < 6) {
    const count = Math.max(days, 1);
    if (count === 1) {
      return abbreviated ? (isPast ? '1 d ago' : 'in 1 d') : isPast ? 'yesterday' : 'tomorrow';
    }
    return abbreviated
      ? isPast
        ? `${count} d ago`
        : `in ${count} d`
      : isPast
        ? `${count} days ago`
        : `in ${count} days`;
  }

  // >= 6 days AND < 27 days → weeks
  if (days < 27) {
    const weeks = Math.round(days / 7);
    if (weeks === 1) {
      return abbreviated ? (isPast ? '1 wk ago' : 'in 1 wk') : isPast ? 'last week' : 'next week';
    }
    return abbreviated
      ? isPast
        ? `${weeks} wk ago`
        : `in ${weeks} wk`
      : isPast
        ? `${weeks} weeks ago`
        : `in ${weeks} weeks`;
  }

  // >= 27 days → Use calendar arithmetic (following GitHub's approach)
  const currentYear = now.getFullYear();
  const currentMonth = now.getMonth();

  // Calculate target date using calendar math
  const targetDate = new Date(date);
  const yearDiff = currentYear - targetDate.getFullYear();
  const monthDiff = currentMonth - targetDate.getMonth();
  const monthsDiff = Math.abs(yearDiff * 12 + monthDiff);

  // If monthsDiff <= 11 → show as months
  if (monthsDiff <= 11) {
    const months = monthsDiff || 1;
    if (months === 1) {
      return abbreviated ? (isPast ? '1 mo ago' : 'in 1 mo') : isPast ? 'last month' : 'next month';
    }
    return abbreviated
      ? isPast
        ? `${months} mo ago`
        : `in ${months} mo`
      : isPast
        ? `${months} months ago`
        : `in ${months} months`;
  }

  // monthsDiff > 11 → show as years
  const years = Math.abs(yearDiff);
  if (years === 1) {
    return abbreviated ? (isPast ? '1 y ago' : 'in 1 y') : isPast ? 'last year' : 'next year';
  }
  return abbreviated
    ? isPast
      ? `${years} y ago`
      : `in ${years} y`
    : isPast
      ? `${years} years ago`
      : `in ${years} years`;
};
