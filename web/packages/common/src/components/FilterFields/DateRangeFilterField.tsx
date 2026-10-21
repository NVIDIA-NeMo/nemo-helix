/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { parseUTCDateForPicker } from '@nemo/common/src/utils/formatDateRange';
import { DatePicker } from '@nvidia/foundations-react-core';
import { useState, type FC } from 'react';

/** Shape used by list APIs for date range (e.g. ModelEntitySearch created_at/updated_at). */
export interface DatetimeFilterValue {
  $gte?: string;
  $lte?: string;
}

export interface DateRangeFilterFieldProps {
  /** Current value (ISO date strings). */
  value?: DatetimeFilterValue;
  /** Called when the user changes the date range. */
  onValueChange: (value: DatetimeFilterValue | undefined) => void;
  /** Optional data-testid for the DatePicker. */
  dataTestId?: string;
}

export const DateRangeFilterField: FC<DateRangeFilterFieldProps> = ({
  value,
  onValueChange,
  dataTestId,
}) => {
  const [open, setOpen] = useState(false);
  // Re-render on open/close so KUI's calendar receives a fresh committed range.
  // Its draft otherwise survives dismissal without Apply.
  const fromDate = value?.$gte ? parseUTCDateForPicker(value.$gte) : undefined;
  const toDate = value?.$lte ? parseUTCDateForPicker(value.$lte) : undefined;

  const handleChange = (dateRange: { from?: Date; to?: Date } | undefined) => {
    if (!dateRange?.from && !dateRange?.to) {
      onValueChange(undefined);
      return;
    }
    // KUI also emits on text-input blur. With timeZone="utc", parsing an
    // unchanged input can copy local clock hours into the UTC date. Ignore
    // date-equivalent emissions so focusing the calendar cannot commit a filter.
    const sameDay = (date: Date | undefined, committed: Date | undefined) =>
      date?.toISOString().slice(0, 10) === committed?.toISOString().slice(0, 10);
    if (sameDay(dateRange?.from, fromDate) && sameDay(dateRange?.to, toDate)) return;

    const $gte = dateRange?.from
      ? new Date(new Date(dateRange.from).setUTCHours(0, 0, 0, 0)).toISOString()
      : undefined;
    const $lte = dateRange?.to
      ? new Date(new Date(dateRange.to).setUTCHours(23, 59, 59, 999)).toISOString()
      : undefined;
    onValueChange({ $gte, $lte });
  };

  return (
    <DatePicker
      data-testid={dataTestId}
      open={open}
      onOpenChange={setOpen}
      kind="range"
      placeholder="yyyy-mm-dd"
      format="yyyy-MM-dd"
      timeZone="utc"
      value={{
        from: fromDate,
        to: toDate,
      }}
      onValueChange={handleChange}
    />
  );
};
