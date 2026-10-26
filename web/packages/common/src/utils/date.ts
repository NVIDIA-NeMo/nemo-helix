// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Calculates the difference in seconds between two ISO date-time strings.
 * @param date1 ISO date-time string or undefined.
 * @param date2 ISO date-time string or undefined.
 * @returns The difference in seconds between the two dates, or undefined if either date is undefined.
 */
export const getDifferenceInMilliseconds = (date1?: string, date2?: string) => {
  if (!date1 || !date2) return undefined;
  const dateValue1 = new Date(date1);
  const dateValue2 = new Date(date2);
  return dateValue2.getTime() - dateValue1.getTime();
};

export const getElapsedSeconds = (startMs: number, endMs: number = Date.now()): number =>
  Math.floor((endMs - startMs) / 1000);

/**
 * Converts a UTC ISO string to a local Date object.
 * Handles ISO strings with or without timezone indicators.
 * @param utcIsoString UTC ISO date-time string (e.g., "2025-11-17T21:53:35.903780" or "2025-11-17T21:53:35.903780Z")
 * @returns Date object in the browser's local timezone, or undefined if the input is invalid
 * @example
 * utcToLocalDate("2025-11-17T21:53:35.903780") // Returns Date object in local time
 * utcToLocalDate("2025-11-17T21:53:35.903780Z") // Returns Date object in local time
 */
export const utcToLocalDate = (utcIsoString?: string): Date | undefined => {
  if (!utcIsoString) return undefined;

  // If the string doesn't have a timezone indicator (Z or +/-HH:MM), append 'Z' to treat it as UTC
  const hasTimezone = /Z|[+-]\d{2}:\d{2}$/.test(utcIsoString);
  const isoString = hasTimezone ? utcIsoString : `${utcIsoString}Z`;

  const date = new Date(isoString);

  // Check if the date is valid
  if (isNaN(date.getTime())) {
    return undefined;
  }

  return date;
};

/** Rendered when a duration is null/undefined. Matches the EM DASH used elsewhere for empty values. */
const EMPTY_DURATION = '—';

/**
 * Formats a duration given in milliseconds into a compact human-friendly string,
 * e.g. `10m 12s 13ms`, `12s 34ms`, or `34ms`.
 *
 * Shows only the units needed: leading and zero-valued units are dropped (so
 * `1h 0m 5s` renders as `1h 5s`), matching {@link formatTimeInSeconds}. Values
 * under one millisecond keep two decimals (`0.34ms`) so span timings are not
 * rounded to zero. Returns an em dash for null/undefined.
 *
 * With `hideMsAboveMinute`, the millisecond component is dropped once the total
 * reaches a minute (`10m 12s`, not `10m 12s 13ms`); sub-minute durations still
 * show ms. Use it where sub-second precision is noise past a minute.
 */
export const formatDurationMs = (
  ms?: number | null,
  { hideMsAboveMinute = false }: { hideMsAboveMinute?: boolean } = {}
): string => {
  if (ms == null) return EMPTY_DURATION;
  if (ms <= 0) return '0ms';
  if (ms < 1) return `${Number(ms.toFixed(2))}ms`;

  const total = Math.round(ms);
  const dropMs = hideMsAboveMinute && total >= 60_000;
  const units: [number, string][] = [
    [Math.floor(total / 3_600_000), 'h'],
    [Math.floor((total % 3_600_000) / 60_000), 'm'],
    [Math.floor((total % 60_000) / 1_000), 's'],
    [total % 1_000, 'ms'],
  ];
  return units
    .filter(([value, unit]) => value > 0 && !(dropMs && unit === 'ms'))
    .map(([value, unit]) => `${value}${unit}`)
    .join(' ');
};

/**
 * Formats the time in seconds into a human-friendly string
 * Only showing the minimum units needed to represent the time
 */
export const formatTimeInSeconds = (seconds?: number) => {
  // Whole seconds only: sub-second (and empty/negative) inputs render nothing here.
  if (!seconds || seconds < 1) return '';
  return formatDurationMs(Math.floor(seconds) * 1000);
};
