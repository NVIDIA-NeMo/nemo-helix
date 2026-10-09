// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { nextIntervalFrom, RelativeTime } from '@nemo/common/src/components/RelativeTime/index';
import { act, render, screen } from '@testing-library/react';

// exact time of the 2024 Apr 8th total eclipse
// above Durango, Mexico at "Point of Greatest Eclipse"
const MOCK_CURRENT_TIME = '2024-04-08T11:07:25.506-07:00';

describe('nextIntervalFrom()', () => {
  beforeEach(() => {
    vi.spyOn(Date, 'now').mockReturnValue(new Date(MOCK_CURRENT_TIME).getTime());
  });

  it('returns null for older events', () => {
    // exactly 3 days in the "past"
    const days_ago = '2024-04-05T11:07:25.506-07:00';
    // too far in the past to set a JS timeout to change it.
    // we'll render "[n] day(s) ago"
    expect(nextIntervalFrom(days_ago)).toBeNull();
  });

  it('returns null for far off future events', () => {
    // exactly 10 days in the "future"
    //              now = '2024-04-08T11:07:25.506-07:00';
    const days_from_now = '2024-04-18T11:07:25.506-07:00';
    // too far in the future to set a JS timeout to change it.
    // we'll render "[n] day(s) ago"
    expect(nextIntervalFrom(days_from_now)).toBeNull();
  });

  it('returns ms until next hour of timestamp', () => {
    // this morning at 01:00am exactly
    const hours_ago = '2024-04-08T01:00:00.000-07:00';
    //          now = '2024-04-08T11:07:25.506-07:00';
    const next_tick = '2024-04-08T12:00:00.000-07:00';
    const expectation = Date.parse(next_tick) - Date.parse(MOCK_CURRENT_TIME);
    expect(nextIntervalFrom(hours_ago)).toBe(expectation);
  });

  it('returns ms until next hour of timestamp for future dates', () => {
    //           now = '2024-04-08T11:07:25.506-07:00';
    const in_3_hours = '2024-04-08T14:00:00.000-07:00';
    const next_tick = '2024-04-08T12:00:00.000-07:00';
    const expectation = Date.parse(next_tick) - Date.parse(MOCK_CURRENT_TIME);
    expect(nextIntervalFrom(in_3_hours)).toBe(expectation);
  });

  it('returns ms until next minute of timestamp', () => {
    // 7 minutes 27 seconds ago; at 11am exactly
    const minutes_ago = '2024-04-08T11:00:00.000-07:00';
    //          now = '2024-04-08T11:07:25.506-07:00';
    const next_tick = '2024-04-08T11:08:00.000-07:00';
    const expectation = Date.parse(next_tick) - Date.parse(MOCK_CURRENT_TIME);
    expect(nextIntervalFrom(minutes_ago)).toBe(expectation);
  });

  it('returns ms until next minute of future timestamp', () => {
    // 17 minutes 35 seconds from now; at 11:25am
    //                 now = '2024-04-08T11:07:25.506-07:00';
    const minutes_from_now = '2024-04-08T11:25:00.000-07:00';
    const next_tick = /*  */ '2024-04-08T11:08:00.000-07:00';
    const expectation = Date.parse(next_tick) - Date.parse(MOCK_CURRENT_TIME);
    expect(nextIntervalFrom(minutes_from_now)).toBe(expectation);
  });

  it('returns null for far future dates (> 1 day)', () => {
    // next month - shouldn't update frequently
    const nextMonth = '2024-05-08T11:07:25.506-07:00';
    expect(nextIntervalFrom(nextMonth)).toBeNull();
  });
});

describe('<RelativeTime />', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(MOCK_CURRENT_TIME));
  });

  it('Treats timezone-naive datetime strings as UTC', () => {
    // Test that a datetime without timezone info is treated as UTC
    const utcDatetimeWithoutTz = '2024-04-08T18:07:25.506'; // No timezone
    const utcDatetimeWithTz = '2024-04-08T18:07:25.506Z'; // Explicit UTC

    // Both should render the same relative time
    const { rerender } = render(<RelativeTime datetime={utcDatetimeWithoutTz} />);
    const firstText = screen.getByRole('time').textContent;

    rerender(<RelativeTime datetime={utcDatetimeWithTz} />);
    const secondText = screen.getByRole('time').textContent;

    expect(firstText).toBe(secondText);
  });

  it('Displays correct relative time', async () => {
    // 2 hours & 52 minutes in the future
    //     TIME = '2024-04-08T11:07:25.506-07:00';
    const event = '2024-04-08T14:00:00.000-07:00';
    render(<RelativeTime datetime={event} />);

    const SECONDS = 1_000; /*ms*/
    const MINUTES = 60 * SECONDS;
    const HOURS = 60 * MINUTES;

    // we're between 2 & 3 hours "right now"
    expect(screen.getByText('in 2 hours')).toBeInTheDocument();

    await act(() => vi.advanceTimersByTimeAsync(2 * HOURS));
    expect(screen.getByText('in 52 minutes')).toBeInTheDocument();

    await act(() => vi.advanceTimersByTimeAsync(52 * MINUTES + 30 * SECONDS));
    expect(screen.getByText('in a minute')).toBeInTheDocument();

    await act(() => vi.advanceTimersByTimeAsync(30 * SECONDS));
    expect(screen.getByText('just now')).toBeInTheDocument();

    await act(() => vi.advanceTimersByTimeAsync(65 * SECONDS));
    expect(screen.getByText('a minute ago')).toBeInTheDocument();

    await act(() => vi.advanceTimersByTimeAsync(2 * MINUTES));
    expect(screen.getByText('3 minutes ago')).toBeInTheDocument();

    await act(() => vi.advanceTimersByTimeAsync(2 * HOURS));
    const hoursLater = screen.getByText('2 hours ago');
    expect(hoursLater).toBeInTheDocument();
  });

  it.each([
    ['2024-04-08T10:12:25.506-07:00', false, 'an hour ago'],
    ['2024-04-08T10:09:25.506-07:00', false, 'an hour ago'],
    ['2024-04-08T10:08:25.506-07:00', false, 'an hour ago'],
    ['2024-04-08T12:05:25.506-07:00', false, 'in an hour'],
    ['2024-04-08T10:09:25.506-07:00', true, '1 h ago'],
    ['2024-04-08T12:05:25.506-07:00', true, 'in 1 h'],
  ])('Rounds 55-59 minutes to one hour (%s, abbreviated=%s)', (event, abbreviated, expected) => {
    render(<RelativeTime datetime={event} abbreviated={abbreviated} />);
    expect(screen.getByText(expected)).toBeInTheDocument();
  });

  it('Displays special labels for specific time periods', () => {
    // Test "yesterday"
    const yesterday = '2024-04-07T11:07:25.506-07:00';
    const { rerender } = render(<RelativeTime datetime={yesterday} />);
    expect(screen.getByText('yesterday')).toBeInTheDocument();

    // Test "last month"
    const lastMonth = '2024-03-08T11:07:25.506-07:00';
    rerender(<RelativeTime datetime={lastMonth} />);
    expect(screen.getByText('last month')).toBeInTheDocument();

    // Test "last year"
    const lastYear = '2023-04-08T11:07:25.506-07:00';
    rerender(<RelativeTime datetime={lastYear} />);
    expect(screen.getByText('last year')).toBeInTheDocument();
  });

  it('Displays abbreviated format when requested', () => {
    // Test "now" instead of "just now"
    const justNow = '2024-04-08T11:07:20.506-07:00'; // 5 seconds ago
    const { rerender } = render(<RelativeTime datetime={justNow} abbreviated />);
    expect(screen.getByText('now')).toBeInTheDocument();

    // Test abbreviated minutes
    const fiveMinutesAgo = '2024-04-08T11:02:25.506-07:00';
    rerender(<RelativeTime datetime={fiveMinutesAgo} abbreviated />);
    expect(screen.getByText('5 min ago')).toBeInTheDocument();

    // Test abbreviated hours
    const twoHoursAgo = '2024-04-08T09:07:25.506-07:00';
    rerender(<RelativeTime datetime={twoHoursAgo} abbreviated />);
    expect(screen.getByText('2 h ago')).toBeInTheDocument();

    // Test abbreviated days
    const threeDaysAgo = '2024-04-05T11:07:25.506-07:00';
    rerender(<RelativeTime datetime={threeDaysAgo} abbreviated />);
    expect(screen.getByText('3 d ago')).toBeInTheDocument();

    // Test abbreviated weeks
    const twoWeeksAgo = '2024-03-25T11:07:25.506-07:00';
    rerender(<RelativeTime datetime={twoWeeksAgo} abbreviated />);
    expect(screen.getByText('2 wk ago')).toBeInTheDocument();

    // Test abbreviated months
    const threeMonthsAgo = '2024-01-08T11:07:25.506-07:00';
    rerender(<RelativeTime datetime={threeMonthsAgo} abbreviated />);
    expect(screen.getByText('3 mo ago')).toBeInTheDocument();

    // Test abbreviated years
    const twoYearsAgo = '2022-04-08T11:07:25.506-07:00';
    rerender(<RelativeTime datetime={twoYearsAgo} abbreviated />);
    expect(screen.getByText('2 y ago')).toBeInTheDocument();
  });

  it('Displays future dates correctly', () => {
    // Test "tomorrow"
    const tomorrow = '2024-04-09T11:07:25.506-07:00';
    const { rerender } = render(<RelativeTime datetime={tomorrow} />);
    expect(screen.getByText('tomorrow')).toBeInTheDocument();

    // Test "next month"
    const nextMonth = '2024-05-08T11:07:25.506-07:00';
    rerender(<RelativeTime datetime={nextMonth} />);
    expect(screen.getByText('next month')).toBeInTheDocument();

    // Test "next year"
    const nextYear = '2025-04-08T11:07:25.506-07:00';
    rerender(<RelativeTime datetime={nextYear} />);
    expect(screen.getByText('next year')).toBeInTheDocument();
  });
});
