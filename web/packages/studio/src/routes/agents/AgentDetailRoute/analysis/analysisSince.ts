// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseISOWithUTCFallback } from '@nemo/common/src/components/RelativeTime/util';

export type SincePreset = 'all' | 'periodic-cursor' | 'day' | 'week' | 'custom';

export const DAY_MS = 24 * 60 * 60 * 1000;

const msAgo = (now: Date, ms: number) => new Date(now.getTime() - ms).toISOString();

/** A `datetime-local` input value for `date` in the viewer's time zone. */
export const toDateTimeLocalValue = (date: Date): string => {
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
};

interface SinceInputs {
  now: Date;
  periodicCursor?: string;
  custom?: string;
}

/** The `since` bound a preset sends, or undefined for the full trace history or an unusable input. */
export const sinceFor = (
  preset: SincePreset,
  { now, periodicCursor, custom }: SinceInputs
): string | undefined => {
  switch (preset) {
    case 'periodic-cursor':
      return periodicCursor ? parseISOWithUTCFallback(periodicCursor).toISOString() : undefined;
    case 'day':
      return msAgo(now, DAY_MS);
    case 'week':
      return msAgo(now, 7 * DAY_MS);
    case 'custom': {
      const date = custom ? new Date(custom) : undefined;
      return date && !Number.isNaN(date.getTime()) && date <= now ? date.toISOString() : undefined;
    }
    case 'all':
      return undefined;
  }
};
