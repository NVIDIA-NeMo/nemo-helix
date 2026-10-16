// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DateTimeFilterControl } from '@nemo/common/src/components/DataView/FilterPanel/DateRangeFilter';
import type { DataViewColumn } from '@nemo/common/src/components/DataView/FilterPanel/types';
import {
  DateRangeFilterField,
  type DatetimeFilterValue,
} from '@nemo/common/src/components/FilterFields/DateRangeFilterField';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

const POPULATED = {
  $gte: '2026-05-10T00:00:00.000Z',
  $lte: '2026-05-20T23:59:59.999Z',
};

function FilterHarness({
  initialValue,
  onCommit,
  control,
}: {
  initialValue?: DatetimeFilterValue;
  onCommit: (value: DatetimeFilterValue | undefined) => void;
  control: string;
}) {
  const [value, setValue] = useState(initialValue);
  const commit = (next: DatetimeFilterValue | undefined) => {
    onCommit(next);
    setValue(next);
  };
  return (
    <>
      {control === 'field' ? (
        <DateRangeFilterField value={value} onValueChange={commit} />
      ) : (
        <DateTimeFilterControl
          column={
            {
              id: 'started_at',
              getFilterValue: () => value,
              setFilterValue: commit,
            } as DataViewColumn
          }
        />
      )}
      <button onClick={() => setValue(POPULATED)}>Replace committed filter</button>
    </>
  );
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.useRealTimers();
});

describe.each(['UTC', 'America/Los_Angeles', 'Asia/Kolkata'])('%s', (timeZone) => {
  describe.each(['field', 'column'])('%s control with the real KUI picker', (control) => {
    it.each([
      undefined,
      POPULATED,
      { $gte: '2026-05-10T14:35:12.000Z', $lte: '2026-05-20T14:35:12.000Z' },
    ])(
      'keeps edits draft until Apply, restores on Cancel and reopening (initial: %j)',
      async (initialValue) => {
        vi.stubEnv('TZ', timeZone);
        vi.setSystemTime(new Date('2026-05-15T12:00:00Z'));
        const user = userEvent.setup();
        const onCommit = vi.fn();
        render(<FilterHarness {...{ initialValue, onCommit, control }} />);
        const [fromInput, toInput] = screen.getAllByRole('textbox');
        const open = () => user.click(fromInput);
        const selectDay = async () => {
          // DayPicker exposes an aria-label on each day. Query it directly rather
          // than computing accessible names for both months on every interaction.
          await user.click(await screen.findByLabelText(/May 25th, 2026/, { selector: 'button' }));
        };
        const expectOriginal = () => {
          expect(fromInput).toHaveValue(initialValue ? '2026-05-10' : '');
          expect(toInput).toHaveValue(initialValue ? '2026-05-20' : '');
        };

        await open();
        await user.click(toInput);
        await selectDay();
        // Moving focus into the calendar must not commit an unchanged text input.
        expect(onCommit).not.toHaveBeenCalled();
        expectOriginal();
        await user.click(screen.getByText('Cancel', { selector: 'button' }));
        expect(onCommit).not.toHaveBeenCalled();
        expectOriginal();
        expect(screen.queryByText('Apply', { selector: 'button' })).not.toBeInTheDocument();

        await user.keyboard('{Escape}');
        await open();
        expect(screen.queryByText('Apply', { selector: 'button' })).not.toBeInTheDocument();
        await selectDay();
        expect(onCommit).not.toHaveBeenCalled();
        await user.click(screen.getByText('Apply', { selector: 'button' }));
        expect(onCommit).toHaveBeenCalledTimes(1);
        expect(onCommit).toHaveBeenLastCalledWith({
          $gte: initialValue ? POPULATED.$gte : '2026-05-25T00:00:00.000Z',
          $lte: '2026-05-25T23:59:59.999Z',
        });
        expect(toInput).toHaveValue('2026-05-25');

        await open();
        expect(screen.queryByText('Apply', { selector: 'button' })).not.toBeInTheDocument();
        await user.click(screen.getByLabelText(/May 28th, 2026/, { selector: 'button' }));
        await user.keyboard('{Escape}');
        await open();
        expect(screen.queryByText('Apply', { selector: 'button' })).not.toBeInTheDocument();
        expect(toInput).toHaveValue('2026-05-25');
        expect(onCommit).toHaveBeenCalledTimes(1);

        await user.keyboard('{Escape}');
        await user.click(screen.getByRole('button', { name: 'Replace committed filter' }));
        await open();
        expect(fromInput).toHaveValue('2026-05-10');
        expect(toInput).toHaveValue('2026-05-20');
        expect(screen.queryByText('Apply', { selector: 'button' })).not.toBeInTheDocument();
        expect(onCommit).toHaveBeenCalledTimes(1);
      },
      // These real-KUI flows exercise several complete calendar sessions. Coverage
      // plus parallel package tests on CI can exceed the default five seconds.
      15_000
    );
  });
});
