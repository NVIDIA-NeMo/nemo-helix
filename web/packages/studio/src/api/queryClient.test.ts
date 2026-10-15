// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

describe('queryClient', () => {
  beforeEach(() => {
    vi.resetModules();
    delete window.__TANSTACK_QUERY_CLIENT__;
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    delete window.__TANSTACK_QUERY_CLIENT__;
  });

  it('exposes the client to the devtools browser extensions in local development', async () => {
    vi.stubEnv('VITE_IS_LOC_ENV', 'true');
    const { queryClient } = await import('@studio/api/queryClient');

    expect(window.__TANSTACK_QUERY_CLIENT__).toBe(queryClient);
  });

  it('keeps the client off window outside local development', async () => {
    vi.stubEnv('VITE_IS_LOC_ENV', 'false');
    await import('@studio/api/queryClient');

    expect(window.__TANSTACK_QUERY_CLIENT__).toBeUndefined();
  });
});
