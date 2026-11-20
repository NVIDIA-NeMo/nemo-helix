// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SwarmEvent } from '@agent-hardener/components/eventTypes';
import { isFeedEvent, levelColor } from '@agent-hardener/components/swarm/MessageFeed';
import { ACCENT } from '@agent-hardener/theme';

const output = (payload: Record<string, unknown>): SwarmEvent => ({ id: 0, event: 'output', payload });

describe('feed output lines by level', () => {
  it('hides detail lines, as the CLI does without --debug', () => {
    expect(isFeedEvent(output({ line: '  · sandbox ah-x', level: 'detail' }))).toBe(false);
    expect(isFeedEvent(output({ line: '✓ Sandbox ready (6.3s)', level: 'ok' }))).toBe(true);
    expect(isFeedEvent({ id: 1, event: 'agent_started', payload: { level: 'detail' } })).toBe(true);
  });

  it('colors ok / warn / fail lines and leaves the rest, and older level-less lines, alone', () => {
    expect(levelColor(output({ level: 'ok' }))).toBe(ACCENT.green);
    expect(levelColor(output({ level: 'warn' }))).toBe(ACCENT.yellow);
    expect(levelColor(output({ level: 'fail' }))).toBe(ACCENT.red);
    expect(levelColor(output({ level: 'section' }))).toBeUndefined();
    expect(levelColor(output({ line: 'sandbox ready' }))).toBeUndefined();
    expect(levelColor({ id: 1, event: 'agent_completed', payload: { level: 'ok' } })).toBeUndefined();
  });
});
