// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describeChatError } from './describeChatError';

// Messages captured from real failures on the agent chat path. The envelope text is written by
// the platform (Fabric server / agents gateway); the inner `Error code: NNN - {...}` comes from the
// openai SDK inside the Fabric runtime, so rules key on that status rather than the provider text.
const RETIRED_MODEL =
  'RuntimeError: internal error: APIStatusError: Error code: 410 - {\'detail\': \'{"type":"about:blank","title":"Gone","status":410,"detail":"The model \\\'nvidia/nemotron-3-nano-30b-a3b\\\' has reached its end of life on 2026-09-01T09:00:00Z and is no longer available."}\\n\'}';

describe('describeChatError', () => {
  it('names the retired model and says what to do', () => {
    const described = describeChatError(RETIRED_MODEL);

    expect(described?.title).toContain('nvidia/nemotron-3-nano-30b-a3b');
    expect(described?.title).toMatch(/retired/i);
    expect(described?.hint).toMatch(/model/i);
  });

  it('still recognises the retired model when the SDK prefixes the HTTP status', () => {
    expect(describeChatError(`502 ${RETIRED_MODEL}`)?.title).toMatch(/retired/i);
  });

  it.each([
    [404, /not found/i],
    [401, /credentials/i],
    [403, /credentials/i],
    [424, /rejected/i],
    [429, /rate/i],
    [503, /provider/i],
  ])('maps an upstream model error %i to a specific title', (status, title) => {
    const described = describeChatError(
      `RuntimeError: internal error: APIStatusError: Error code: ${status} - {'detail': 'x'}`
    );

    expect(described?.title).toMatch(title);
    expect(described?.hint).not.toBe('');
  });

  it('explains a runtime that failed to start and keeps the adapter detail out of the title', () => {
    const described = describeChatError(
      'Fabric runtime startup failed: adapter nvidia.fabric.a2ui-redteam does not support telemetry.providers value relay'
    );

    expect(described?.title).toMatch(/failed to start/i);
    expect(described?.hint).toMatch(/logs/i);
  });

  it('turns the gateway connect/timeout message, including its empty-reason form, into a status hint', () => {
    const empty = describeChatError('Could not connect to agent: ');
    const refused = describeChatError('Could not connect to agent: All connection attempts failed');

    expect(empty).toEqual(refused);
    expect(empty?.title).toMatch(/did not respond|no response/i);
    expect(empty?.hint).toMatch(/deployment/i);
  });

  it('explains a bare 502 with no body as a proxy or network problem', () => {
    const described = describeChatError('502 status code (no body)');

    expect(described?.title).toMatch(/502/);
    expect(described?.hint).toMatch(/proxy|network|reachable/i);
  });

  it('returns nothing for messages it does not know, so the raw text and generic banner stand', () => {
    expect(describeChatError('Something nobody mapped')).toBeUndefined();
    expect(describeChatError('Agent returned 502: {"detail":"boom"}')).toBeUndefined();
    expect(describeChatError('')).toBeUndefined();
  });
});
