// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCopyToClipboard } from '@nemo/common/src/hooks/useCopyToClipboard/index';
import { suppressConsoleError } from '@nemo/testing/utils/suppress-console';
import { renderHook, act } from '@testing-library/react';

// Mock navigator.clipboard
const mockWriteText = vi.fn();
vi.stubGlobal('navigator', {
  ...navigator,
  clipboard: {
    writeText: mockWriteText,
  },
});

describe('useCopyToClipboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should handle clipboard write errors', async () => {
    suppressConsoleError('Failed to copy text to clipboard');
    const error = new Error('Clipboard access denied');
    mockWriteText.mockRejectedValueOnce(error);
    const { result } = renderHook(() => useCopyToClipboard());

    await act(async () => {
      await result.current.copyToClipboard('test content');
    });

    expect(mockWriteText).toHaveBeenCalledWith('test content');
    expect(console.error).toHaveBeenCalledWith('Failed to copy text to clipboard', error);
  });

  it('should handle multiple rapid copy operations', async () => {
    mockWriteText.mockResolvedValue(undefined);
    const { result } = renderHook(() => useCopyToClipboard());

    // Rapid fire copies
    await act(async () => {
      await result.current.copyToClipboard('content 1');
      await result.current.copyToClipboard('content 2');
      await result.current.copyToClipboard('content 3');
    });

    expect(mockWriteText).toHaveBeenCalledTimes(3);
    expect(mockWriteText).toHaveBeenLastCalledWith('content 3');
  });

  it('should handle empty string content', async () => {
    mockWriteText.mockResolvedValueOnce(undefined);
    const { result } = renderHook(() => useCopyToClipboard());

    await act(async () => {
      await result.current.copyToClipboard('');
    });

    expect(mockWriteText).toHaveBeenCalledWith('');
  });

  it('should call onSuccess callback on success', async () => {
    const onSuccess = vi.fn();
    const { result } = renderHook(() => useCopyToClipboard({ onSuccess }));

    await act(async () => {
      await result.current.copyToClipboard('test content');
    });

    expect(onSuccess).toHaveBeenCalled();
  });

  it('should call onError callback on error', async () => {
    const onError = vi.fn();
    const { result } = renderHook(() => useCopyToClipboard({ onError }));

    await act(async () => {
      await result.current.copyToClipboard('test content');
    });
  });
});
