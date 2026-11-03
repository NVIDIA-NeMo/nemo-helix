// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  describeChatError,
  GENERIC_CHAT_ERROR,
  humanizeChatError,
} from '@nemo/common/src/components/AssistantChat/describeChatError';

// Messages captured from real failures on the agent chat path. The envelope text is written by
// the platform (Fabric server / agents gateway); the inner `APIStatusError: Error code: NNN` comes
// from the openai SDK inside the Fabric runtime, so rules key on that status rather than the
// provider text.
const RETIRED_MODEL =
  'RuntimeError: internal error: APIStatusError: Error code: 410 - {\'detail\': \'{"type":"about:blank","title":"Gone","status":410,"detail":"The model \\\'nvidia/nemotron-3-nano-30b-a3b\\\' has reached its end of life on 2026-09-01T09:00:00Z and is no longer available."}\\n\'}';
const RUNTIME_START_FAILED =
  'Fabric runtime startup failed: adapter nvidia.fabric.a2ui-redteam does not support telemetry.providers value relay';

const upstream = (status: number) =>
  `RuntimeError: internal error: APIStatusError: Error code: ${status} - {'detail': 'x'}`;

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
    [404, /was not found/i],
    [401, /credentials/i],
    [403, /credentials/i],
    [424, /rejected the request/i],
    [429, /rate limiting/i],
    [503, /unavailable/i],
  ])('gives upstream model error %i its own title', (status, title) => {
    expect(describeChatError(upstream(status))?.title).toMatch(title);
  });

  it.each([400, 409, 422])('has no rule for upstream status %i', (status) => {
    expect(describeChatError(upstream(status))).toBeUndefined();
  });

  it('explains a runtime that failed to start', () => {
    const described = describeChatError(RUNTIME_START_FAILED);

    expect(described?.title).toMatch(/failed to start/i);
    expect(described?.hint).toMatch(/logs/i);
  });

  it('does not blame the model provider when a platform failure quotes a status code', () => {
    const startFailure = `${RUNTIME_START_FAILED} (APIStatusError: Error code: 401)`;
    const connectFailure = 'Could not connect to agent: APIStatusError: Error code: 503';

    expect(describeChatError(startFailure)?.title).toMatch(/failed to start/i);
    expect(describeChatError(connectFailure)?.title).toMatch(/did not respond/i);
  });

  it('turns the gateway connect/timeout message, including its empty-reason form, into a status hint', () => {
    const empty = describeChatError('Could not connect to agent: ');
    const refused = describeChatError('Could not connect to agent: All connection attempts failed');

    expect(empty).toEqual(refused);
    expect(empty?.title).toMatch(/did not respond/i);
    expect(empty?.hint).toMatch(/deployment/i);
  });

  it('explains a bare 502 with no body as a proxy or network problem', () => {
    const described = describeChatError('502 status code (no body)');

    expect(described?.title).toMatch(/502/);
    expect(described?.hint).toMatch(/proxy|network|reachable/i);
  });

  it('ignores a platform phrase or status code that only appears mid-message', () => {
    expect(describeChatError('The tool said: Fabric runtime startup failed')).toBeUndefined();
    expect(describeChatError("The user wrote 'Error code: 410' in the prompt")).toBeUndefined();
  });

  it('ignores a status that only appears inside quoted detail', () => {
    const quoted = 'Agent returned 502: {"detail":"log: APIStatusError: Error code: 401"}';
    const wrapped =
      "RuntimeError: internal error: ValueError: saw 'RuntimeError: internal error: APIStatusError: Error code: 401 - x' in a log";

    expect(describeChatError(quoted)).toBeUndefined();
    expect(describeChatError(wrapped)).toBeUndefined();
  });

  it('returns nothing for messages it does not know', () => {
    expect(describeChatError('Something nobody mapped')).toBeUndefined();
    expect(describeChatError('Agent returned 502: {"detail":"boom"}')).toBeUndefined();
    expect(describeChatError('')).toBeUndefined();
  });
});

describe('humanizeChatError', () => {
  it('joins the description title and hint', () => {
    expect(humanizeChatError(RETIRED_MODEL, describeChatError)).toMatch(
      /has been retired\. Update the agent/
    );
  });

  it('falls back to the generic line without a describer or for an unknown message', () => {
    expect(humanizeChatError(RETIRED_MODEL)).toBe(GENERIC_CHAT_ERROR);
    expect(humanizeChatError('Something nobody mapped', describeChatError)).toBe(
      GENERIC_CHAT_ERROR
    );
  });
});
