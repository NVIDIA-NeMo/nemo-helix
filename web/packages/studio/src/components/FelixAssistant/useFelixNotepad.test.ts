// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NOTEPAD_MS, useFelixNotepad } from '@studio/components/FelixAssistant/useFelixNotepad';
import { act, renderHook } from '@testing-library/react';

const renderNotepad = (initialKey: string | null) =>
  renderHook((requestKey: string | null) => useFelixNotepad(requestKey), {
    initialProps: initialKey,
  });

const advance = (ms: number) => {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
};

describe('useFelixNotepad', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('stays away when nothing is waiting on the user', () => {
    const { result } = renderNotepad(null);

    expect(result.current).toBe(false);
  });

  it('comes out as soon as a request arrives and goes away after 10 seconds', () => {
    const { result, rerender } = renderNotepad(null);

    rerender('request-1');
    expect(result.current).toBe(true);

    advance(NOTEPAD_MS - 1);
    expect(result.current).toBe(true);
    advance(1);
    expect(result.current).toBe(false);
    expect(NOTEPAD_MS).toBe(10_000);
  });

  it('does not come back for the same request once it has gone away', () => {
    const { result, rerender } = renderNotepad('request-1');

    advance(NOTEPAD_MS);
    rerender('request-1');

    expect(result.current).toBe(false);
  });

  it('puts the notepad away as soon as the request is answered', () => {
    const { result, rerender } = renderNotepad('request-1');
    expect(result.current).toBe(true);

    rerender(null);

    expect(result.current).toBe(false);
  });

  it('starts a fresh 10 seconds for a different request', () => {
    const { result, rerender } = renderNotepad('request-1');

    advance(NOTEPAD_MS);
    expect(result.current).toBe(false);

    rerender('request-2');
    expect(result.current).toBe(true);
    advance(NOTEPAD_MS - 1);
    expect(result.current).toBe(true);
    advance(1);
    expect(result.current).toBe(false);
  });
});
