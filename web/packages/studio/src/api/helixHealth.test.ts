// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { checkHelixHealth, getHelixHealthUrl } from '@studio/api/helixHealth';
import { server } from '@studio/mocks/node';
import { http, HttpResponse } from 'msw';

describe('checkHelixHealth', () => {
  it('probes /health/ready on the platform base URL', () => {
    expect(getHelixHealthUrl()).toBe('http://localhost:8080/health/ready');
  });

  it('resolves "ready" on 200', async () => {
    await expect(checkHelixHealth()).resolves.toBe('ready');
  });

  it('resolves "not-ready" when the platform answers 503 with its not_ready body', async () => {
    server.use(
      http.get(getHelixHealthUrl(), () =>
        HttpResponse.json({ detail: { status: 'not_ready' } }, { status: 503 })
      )
    );
    await expect(checkHelixHealth()).resolves.toBe('not-ready');
  });

  it('resolves "unreachable" on a 503 that is not from the platform (e.g. a proxy)', async () => {
    server.use(
      http.get(getHelixHealthUrl(), () => new HttpResponse('Bad Gateway', { status: 503 }))
    );
    await expect(checkHelixHealth()).resolves.toBe('unreachable');
  });

  it('resolves "unreachable" on a network error', async () => {
    server.use(http.get(getHelixHealthUrl(), () => HttpResponse.error()));
    await expect(checkHelixHealth()).resolves.toBe('unreachable');
  });

  it('resolves "unreachable" on an unexpected non-200 status', async () => {
    server.use(http.get(getHelixHealthUrl(), () => new HttpResponse(null, { status: 404 })));
    await expect(checkHelixHealth()).resolves.toBe('unreachable');
  });
});
