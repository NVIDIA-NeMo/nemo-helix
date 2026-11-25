// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describeApiError, toastApiError } from '@agent-hardener/api/apiError';
import { AxiosError, type AxiosResponse } from 'axios';

const rejected = (status: number, data: unknown) =>
  new AxiosError('Request failed', 'ERR_BAD_REQUEST', undefined, undefined, {
    status,
    data,
  } as AxiosResponse);
const unreachable = () => new AxiosError('Network Error', 'ERR_NETWORK');

it("shows the backend's detail instead of axios' generic message", () => {
  expect(describeApiError('Failed to save', rejected(422, { detail: 'secret x is missing' }))).toBe(
    'Failed to save: secret x is missing'
  );
});

it('falls back to the status line when the backend sent no detail', () => {
  expect(describeApiError('Failed to save', rejected(502, {}))).toMatch(/^Failed to save: 502/);
});

it('keeps the message of a non-API error and tolerates non-Error throws', () => {
  expect(describeApiError('Upload', new Error('file too large'))).toBe('Upload: file too large');
  expect(describeApiError('Upload', 'boom')).toBe('Upload: boom');
});

it('toasts the described error', () => {
  const toast = { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() };
  toastApiError(toast, 'Failed to start')(rejected(409, { detail: 'already running' }));
  expect(toast.error).toHaveBeenCalledWith('Failed to start: already running');
});

it("says the service was unreachable, using Studio's wording, when no response came back", () => {
  expect(describeApiError('Validation failed', unreachable())).toMatch(/^Validation failed: .*ERR_NETWORK/);
});
