/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { isUnroutableHost, resolveBrowserBaseUrl } from '@nemo/sdk/src/utils/url';

describe('isUnroutableHost', () => {
  it.each([
    'http://0.0.0.0:8080',
    'https://0.0.0.0/',
    'http://[::]:8080',
    'http://[0:0:0:0:0:0:0:0]:8080',
    'http://nemo-helix-api:8080',
    'http://nemo-helix-api.default.svc.cluster.local:8080',
    'http://nemo-helix-api.default.svc.cluster.local.:8080',
  ])('treats %s as unroutable', (url) => {
    expect(isUnroutableHost(url)).toBe(true);
  });

  it.each([
    'http://localhost:8080',
    'https://example.com',
    'http://127.0.0.1:8080',
    'http://[::1]:8080',
    'https://api.svc.example.com',
    'http://nemo-helix-api.default.svc:8080',
  ])('treats %s as routable', (url) => {
    expect(isUnroutableHost(url)).toBe(false);
  });

  it('returns false for invalid URLs', () => {
    expect(isUnroutableHost('not a url')).toBe(false);
    expect(isUnroutableHost('')).toBe(false);
  });
});

describe('resolveBrowserBaseUrl', () => {
  it('returns the env value when routable', () => {
    expect(resolveBrowserBaseUrl('http://localhost:8080')).toBe('http://localhost:8080');
    expect(resolveBrowserBaseUrl('https://api.svc.example.com')).toBe(
      'https://api.svc.example.com'
    );
  });

  it('falls back to window.location.origin when env value is not browser-routable', () => {
    expect(resolveBrowserBaseUrl('http://0.0.0.0:8080')).toBe(window.location.origin);
    expect(resolveBrowserBaseUrl('http://nemo-helix-api:8080')).toBe(window.location.origin);
  });

  it('falls back to window.location.origin when env value is empty', () => {
    expect(resolveBrowserBaseUrl(undefined)).toBe(window.location.origin);
    expect(resolveBrowserBaseUrl('')).toBe(window.location.origin);
  });
});
