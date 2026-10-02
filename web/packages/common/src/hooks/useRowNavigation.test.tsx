// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { renderHook } from '@testing-library/react';
import React from 'react';

const navigate = vi.fn();

vi.mock('react-router', () => ({
  useNavigate: () => navigate,
}));

const mouseEvent = (init: Partial<React.MouseEvent> = {}) =>
  ({ metaKey: false, ctrlKey: false, button: 0, ...init }) as React.MouseEvent;

describe('useRowNavigation', () => {
  let openSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    navigate.mockClear();
    openSpy = vi.spyOn(window, 'open').mockImplementation(() => null);
  });

  afterEach(() => {
    openSpy.mockRestore();
  });

  it('should navigate in place on a plain click', () => {
    const { result } = renderHook(() => useRowNavigation());

    result.current(mouseEvent(), '/jobs/my-job');

    expect(navigate).toHaveBeenCalledWith('/jobs/my-job');
    expect(openSpy).not.toHaveBeenCalled();
  });

  it('should open a new tab on cmd+click', () => {
    const { result } = renderHook(() => useRowNavigation());

    result.current(mouseEvent({ metaKey: true }), '/jobs/my-job');

    expect(openSpy).toHaveBeenCalledWith('/jobs/my-job', '_blank', 'noopener,noreferrer');
    expect(navigate).not.toHaveBeenCalled();
  });

  it('should open a new tab on ctrl+click', () => {
    const { result } = renderHook(() => useRowNavigation());

    result.current(mouseEvent({ ctrlKey: true }), '/jobs/my-job');

    expect(openSpy).toHaveBeenCalledWith('/jobs/my-job', '_blank', 'noopener,noreferrer');
    expect(navigate).not.toHaveBeenCalled();
  });

  it('should navigate in place when there is no event', () => {
    const { result } = renderHook(() => useRowNavigation());

    result.current(undefined, '/jobs/my-job');

    expect(navigate).toHaveBeenCalledWith('/jobs/my-job');
    expect(openSpy).not.toHaveBeenCalled();
  });

  it('should open a new tab on middle click', () => {
    const { result } = renderHook(() => useRowNavigation());

    result.current(mouseEvent({ button: 1 }), '/jobs/my-job');

    expect(openSpy).toHaveBeenCalledWith('/jobs/my-job', '_blank', 'noopener,noreferrer');
    expect(navigate).not.toHaveBeenCalled();
  });
});
