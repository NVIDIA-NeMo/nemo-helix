// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Formats a date range (start and/or end dates) into a readable string.
 * If only start date is provided, shows "from [date]"
 * If only end date is provided, shows "until [date]"
 * If both are provided, shows "[start] — [end]"
 */
export const formatDateRange = (startDate?: string | number, endDate?: string | number) => {
  // Bounds are UTC instants; render the UTC day so the tag matches the picker.
  const formatDate = (date: string | number) =>
    new Date(date).toLocaleDateString(undefined, { timeZone: 'UTC' });

  if (startDate && endDate) {
    return `${formatDate(startDate)} — ${formatDate(endDate)}`;
  } else if (startDate) {
    return formatDate(startDate);
  } else if (endDate) {
    return formatDate(endDate);
  }

  return '';
};

/**
 * Creates a Date object from a date string for use with KUI DatePicker with timeZone="utc".
 *
 * This function extracts the date portion from an ISO string and creates a Date object
 * at midnight UTC for that date. This ensures consistent display across all timezones
 * when the DatePicker is in UTC mode.
 *
 * Without this, dates can shift by one day for users in timezones significantly
 * different from UTC (e.g., selecting Oct 16 might display as Oct 15).
 *
 * @param dateString - ISO 8601 date string (e.g., "2024-10-16T00:00:00.000Z")
 * @returns Date object at midnight UTC for the date portion of the input
 *
 * @example
 * // User in PST (UTC-8) timezone
 * const date = parseUTCDateForPicker("2024-10-16T00:00:00.000Z");
 * // Returns a Date that DatePicker with timeZone="utc" will display as 2024-10-16
 */
export const parseUTCDateForPicker = (dateString: string): Date => {
  // Extract just the date portion (YYYY-MM-DD) from the ISO string
  // This avoids timezone interpretation issues
  const datePart = dateString.split('T')[0];
  const [year, month, day] = datePart.split('-').map(Number);

  // Create a Date at midnight UTC using the extracted components
  // This ensures the DatePicker with timeZone="utc" displays the correct date
  return new Date(Date.UTC(year, month - 1, day, 0, 0, 0, 0));
};
