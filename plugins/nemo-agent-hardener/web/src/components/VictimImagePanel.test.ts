// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { buildTag } from '@agent-hardener/components/VictimImagePanel';

describe('buildTag', () => {
  it('replaces a digest pin with the latest tag on the same repository', () => {
    expect(buildTag('reg.local:5000/team/victim@sha256:abc123')).toBe('reg.local:5000/team/victim:latest');
  });

  it('keeps an existing tag and an empty value as they are', () => {
    expect(buildTag('reg/victim:v2')).toBe('reg/victim:v2');
    expect(buildTag('')).toBe('');
  });
});
