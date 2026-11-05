// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Turns a dateString like "2024-02-29T02:27:13.509827", or a timestamp in milliseconds like 1721151036524,
 * into a formatted date string. Expects a UTC date string or normalizes input to UTC if a date string is provided without a timezone.
 *
 * If `includeDate` is true, it will include the date like "02/29/24 02:27:13 PM"
 * Otherwise, it will only include the time like "02:27:12 PM"
 */
export function formatDateTime(dateValue: string | number, includeDate: boolean = true) {
  const tzRegex = /Z|[+-]\d{2}:\d{2}$/;
  const normalized =
    typeof dateValue === 'string' && !tzRegex.test(dateValue) ? dateValue + 'Z' : dateValue;
  const date = new Date(normalized);
  const userLocale = navigator.language;

  const options: Intl.DateTimeFormatOptions = {
    ...(includeDate && {
      year: '2-digit',
      month: '2-digit',
      day: '2-digit',
    }),
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  };

  const formatter = new Intl.DateTimeFormat(userLocale, options);
  return formatter.format(date).replace(',', '');
}

/**
 * Returns the elapsed time between the given `startDate` and `endDate` as a string, formatted like hh:mm:ss
 *
 * @param startDate start date of elapsed time range
 * @param endDate end date of elapsed time range
 */
export const formatElapsedTime = (startDate: Date, endDate: Date) => {
  const diff = Math.abs(endDate.getTime() - startDate.getTime());
  const diffInSeconds = Math.floor(diff / 1000);

  const hours = Math.floor(diffInSeconds / 3600);
  const minutes = Math.floor((diffInSeconds % 3600) / 60);
  const seconds = diffInSeconds % 60;

  // Ensure each chunk of the formatted string contains two digits
  const formatDigit = (digit: number) => digit.toString().padStart(2, '0');

  return `${formatDigit(hours)}:${formatDigit(minutes)}:${formatDigit(seconds)}`;
};
