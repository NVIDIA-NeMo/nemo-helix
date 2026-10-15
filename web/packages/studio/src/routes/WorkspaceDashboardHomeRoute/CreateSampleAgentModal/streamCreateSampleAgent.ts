// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  SampleAgentStreamEventKind,
  type CreateSampleAgentRequest,
  type SampleAgentResponse,
  type SampleAgentStreamEvent,
} from '@nemo/sdk/generated/agents/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { asRecord } from '@studio/util/guards';
import { readLineDelimitedStream } from '@studio/util/lineStream';

const SAMPLE_AGENT_PATH = '/apis/agents/v2/sample-agent';

const FRAME_KINDS: readonly string[] = Object.values(SampleAgentStreamEventKind);

/** Parses one NDJSON line; unrecognised lines are dropped. */
export const parseSampleAgentFrame = (line: string): SampleAgentStreamEvent | undefined => {
  const trimmed = line.trim();
  if (!trimmed) return undefined;

  let decoded: unknown;
  try {
    decoded = JSON.parse(trimmed);
  } catch {
    return undefined;
  }

  const frame = asRecord(decoded);
  if (!frame || typeof frame.kind !== 'string' || !FRAME_KINDS.includes(frame.kind)) {
    return undefined;
  }
  return frame as unknown as SampleAgentStreamEvent;
};

/** FastAPI's `detail` (a string or pydantic error list). Other bodies, e.g. proxy HTML, are ignored. */
const detailFromErrorBody = (body: string): string | undefined => {
  let decoded: unknown;
  try {
    decoded = JSON.parse(body);
  } catch {
    return undefined;
  }
  const detail = asRecord(decoded)?.detail;
  if (typeof detail === 'string') return detail;
  if (!Array.isArray(detail)) return undefined;
  const messages = detail.flatMap((item) => {
    const msg = asRecord(item)?.msg;
    return typeof msg === 'string' ? [msg] : [];
  });
  return messages.length ? messages.join(' ') : undefined;
};

const messageForStatus = (status: number): string =>
  status >= 500
    ? `The platform couldn't handle the request (status ${status}). Try again in a moment.`
    : `The request was rejected (status ${status}).`;

/** Errors after the stream opens arrive as an `error` frame inside a 2xx response. */
export const streamCreateSampleAgent = async (
  request: CreateSampleAgentRequest,
  accessToken: string | undefined,
  signal: AbortSignal
): Promise<SampleAgentResponse> => {
  let response: Response;
  try {
    response = await fetch(`${PLATFORM_BASE_URL}${SAMPLE_AGENT_PATH}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        'X-Source': 'NeMo Studio',
      },
      body: JSON.stringify(request),
      signal,
    });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error("Couldn't reach the platform. Check your connection and try again.");
  }

  if (!response.ok) {
    const body = await response.text();
    throw new Error(detailFromErrorBody(body) ?? messageForStatus(response.status));
  }
  if (!response.body) throw new Error('The sample setup response was empty.');

  let result: SampleAgentResponse | undefined;
  let failure: string | undefined;
  await readLineDelimitedStream(response.body, (line) => {
    const frame = parseSampleAgentFrame(line);
    if (!frame || result || failure) return;
    if (frame.kind === 'error') failure = frame.message ?? 'Sample setup failed.';
    else if (frame.kind === 'done' && frame.result) result = frame.result;
  });

  if (failure) throw new Error(failure);
  if (!result)
    throw new Error('The connection closed before setup finished. Try again to finish it.');
  return result;
};
