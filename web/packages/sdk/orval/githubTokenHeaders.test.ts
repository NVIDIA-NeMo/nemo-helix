// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest';
import { getGithubTokenHeaders } from './githubTokenHeaders';

describe('getGithubTokenHeaders', () => {
  it.each(['github.com', 'api.github.com', 'raw.githubusercontent.com'])(
    'attaches the token for default GitHub host %s',
    (host) => {
      expect(
        getGithubTokenHeaders(new URL(`https://${host}/NVIDIA-NeMo/nemo-helix`), 'token')
      ).toEqual({
        Authorization: 'Bearer token',
      });
    }
  );

  it('does not attach the token to non-GitHub hosts', () => {
    expect(
      getGithubTokenHeaders(new URL('https://example.com/openapi.yaml'), 'token')
    ).toBeUndefined();
  });

  it('does not attach a header when no token is configured', () => {
    expect(
      getGithubTokenHeaders(
        new URL('https://raw.githubusercontent.com/NVIDIA-NeMo/nemo-helix'),
        undefined
      )
    ).toBeUndefined();
  });

  it('supports an explicit host allowlist', () => {
    expect(
      getGithubTokenHeaders(
        new URL('https://github.example.com/spec.yaml'),
        'token',
        'github.example.com'
      )
    ).toEqual({
      Authorization: 'Bearer token',
    });
  });
});
