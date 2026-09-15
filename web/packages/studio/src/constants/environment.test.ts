// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

describe('environment', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('preserves the case of a case-sensitive env var like AUTH_CLIENT_ID', async () => {
    vi.stubEnv('VITE_AUTH_CLIENT_ID', 'MixedCaseClientId');
    const { AUTH_CLIENT_ID } = await import('./environment');
    expect(AUTH_CLIENT_ID).toBe('MixedCaseClientId');
  });

  it.each([
    ['true', true],
    ['True', true],
    ['false', false],
  ])('normalizes VITE_TELEMETRY_ENABLED=%s to %s regardless of case', async (value, expected) => {
    vi.stubEnv('VITE_TELEMETRY_ENABLED', value);
    const { TELEMETRY_ENABLED } = await import('./environment');
    expect(TELEMETRY_ENABLED).toBe(expected);
  });
});
