// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ElapsedTime } from '@nemo/common/src/components/ElapsedTime';
import { act, render, screen } from '@testing-library/react';

describe('ElapsedTime', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-10-16T12:00:00Z'));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('shows the fixed time from start to end once stopped', () => {
    render(<ElapsedTime start="2026-10-16T10:00:00Z" end="2026-10-16T11:30:05Z" running={false} />);

    expect(screen.getByText('1h 30m 5s')).toBeInTheDocument();
  });

  it('shows whole seconds once stopped, like the live count', () => {
    const { rerender } = render(
      <ElapsedTime start="2026-10-16T10:00:00Z" end="2026-10-16T10:00:30.500Z" running={false} />
    );
    expect(screen.getByText('30s')).toBeInTheDocument();

    rerender(
      <ElapsedTime start="2026-10-16T10:00:00Z" end="2026-10-16T10:00:00Z" running={false} />
    );
    expect(screen.getByText('0s')).toBeInTheDocument();
  });

  it('counts up live from start while running, starting at the real elapsed time', () => {
    render(<ElapsedTime start="2026-10-13T12:00:00Z" end="2026-10-13T12:00:01Z" running />);

    expect(screen.getByText('72h')).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(5_000);
    });

    expect(screen.getByText('72h 5s')).toBeInTheDocument();
  });

  it('stops counting and shows the fixed time once it stops running', () => {
    const { rerender } = render(
      <ElapsedTime start="2026-10-16T11:00:00Z" end="2026-10-16T11:59:00Z" running />
    );
    expect(screen.getByText('1h')).toBeInTheDocument();

    rerender(
      <ElapsedTime start="2026-10-16T11:00:00Z" end="2026-10-16T11:59:00Z" running={false} />
    );
    act(() => {
      vi.advanceTimersByTime(5_000);
    });

    expect(screen.getByText('59m')).toBeInTheDocument();
  });

  it('shows an em dash without a start, or without an end once stopped', () => {
    const { rerender } = render(<ElapsedTime running />);
    expect(screen.getByText('—')).toBeInTheDocument();

    rerender(<ElapsedTime start="2026-10-16T10:00:00Z" running={false} />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });
});
