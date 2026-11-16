// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toRequestsCsv } from '@agent-hardener/routes/AgentHardenerManifestDetailRoute/utils';
import { describe, expect, it } from 'vitest';

describe('toRequestsCsv', () => {
  it('keeps the header for an empty suite', () => {
    expect(toRequestsCsv([])).toBe('tool,payload,label,rationale,persona');
  });

  it('writes payloads verbatim so the file re-uploads unchanged', () => {
    expect(toRequestsCsv([{ tool: 'calc', payload: '=1+1, then -2' }])).toBe(
      'tool,payload,label,rationale,persona\ncalc,"=1+1, then -2",,,'
    );
  });
});
