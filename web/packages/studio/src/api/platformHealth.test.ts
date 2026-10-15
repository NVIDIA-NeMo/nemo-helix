// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { checkPlatformHealth, getPlatformHealthUrl } from '@studio/api/platformHealth';
import { server } from '@studio/mocks/node';
import { http, HttpResponse } from 'msw';

describe('checkPlatformHealth', () => {
  it('probes /health/live on the platform base URL', () => {
    expect(getPlatformHealthUrl()).toBe('http://localhost:8080/health/live');
  });

  it('resolves "live" on the platform liveness response', async () => {
    await expect(checkPlatformHealth()).resolves.toBe('live');
  });

  it('resolves "unreachable" on a 200 that is not the platform liveness body (e.g. SPA fallback)', async () => {
    server.use(
      http.get(getPlatformHealthUrl(), () =>
        HttpResponse.html('<!doctype html><html><body>Studio</body></html>')
      )
    );
    await expect(checkPlatformHealth()).resolves.toBe('unreachable');
  });

  it('resolves "unreachable" when the liveness endpoint answers 503', async () => {
    server.use(
      http.get(getPlatformHealthUrl(), () =>
        HttpResponse.json({ detail: { status: 'not_ready' } }, { status: 503 })
      )
    );
    await expect(checkPlatformHealth()).resolves.toBe('unreachable');
  });

  it('resolves "unreachable" on a 503 that is not from the platform (e.g. a proxy)', async () => {
    server.use(
      http.get(getPlatformHealthUrl(), () => new HttpResponse('Bad Gateway', { status: 503 }))
    );
    await expect(checkPlatformHealth()).resolves.toBe('unreachable');
  });

  it('resolves "unreachable" on a network error', async () => {
    server.use(http.get(getPlatformHealthUrl(), () => HttpResponse.error()));
    await expect(checkPlatformHealth()).resolves.toBe('unreachable');
  });

  it('resolves "unreachable" on an unexpected non-200 status', async () => {
    server.use(http.get(getPlatformHealthUrl(), () => new HttpResponse(null, { status: 404 })));
    await expect(checkPlatformHealth()).resolves.toBe('unreachable');
  });
});
