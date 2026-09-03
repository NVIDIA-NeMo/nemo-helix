// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ingestAtif,
  ingestChatCompletion,
  ingestOtlpTraces,
  ingestSpans,
} from '@nemo/sdk/generated/platform/ingest';
import { detectTraceFormat } from '@studio/components/ImportTracesModal/detectTraceFormat';
import {
  ingestTraceFile,
  ingestTraceFiles,
  readTraceFile,
  type SelectedTraceFile,
} from '@studio/components/ImportTracesModal/ingestTraceFiles';

vi.mock('@nemo/sdk/generated/platform/ingest', () => ({
  ingestAtif: vi.fn(),
  ingestSpans: vi.fn(),
  ingestChatCompletion: vi.fn(),
  ingestOtlpTraces: vi.fn(),
}));

const atifMock = vi.mocked(ingestAtif);
const spansMock = vi.mocked(ingestSpans);
const chatMock = vi.mocked(ingestChatCompletion);
const otlpMock = vi.mocked(ingestOtlpTraces);

const workspace = 'default';

const trajectory = (name: string) => ({
  schema_version: 'ATIF-v1.5',
  agent: { name },
  steps: [],
});

const span = (id: string, attributes?: Record<string, unknown>) => ({
  span_id: id,
  trace_id: 'trace-1',
  started_at: '2026-01-01T00:00:00Z',
  ...(attributes ? { attributes } : {}),
});

/** Builds a picked file the way the modal does, through real detection. */
const selected = (label: string, value: unknown): SelectedTraceFile => {
  const text = JSON.stringify(value);
  return {
    id: label,
    label,
    file: new File([text], label, { type: 'application/json' }),
    detection: detectTraceFormat(label, text),
  };
};

beforeEach(() => {
  vi.resetAllMocks();
});

describe('readTraceFile', () => {
  it('reads a JSON file and detects its format', async () => {
    const file = new File([JSON.stringify(trajectory('a'))], 'trace.json');
    const read = await readTraceFile(file);

    expect(read.label).toBe('trace.json');
    expect(read.detection.format).toBe('atif');
  });

  it('never reads a protobuf as text', async () => {
    const file = new File(['binary'], 'export.binpb');
    const text = vi.spyOn(file, 'text');

    const read = await readTraceFile(file);

    expect(read.detection.format).toBe('otlp-protobuf');
    expect(text).not.toHaveBeenCalled();
  });
});

describe('ingestTraceFile', () => {
  it('sends ATIF trajectories one at a time and reports the agents seen', async () => {
    const outcome = await ingestTraceFile(
      selected('batch.json', [trajectory('agent-a'), trajectory('agent-b')]),
      { workspace }
    );

    expect(atifMock).toHaveBeenCalledTimes(2);
    expect(outcome.agents).toEqual(['agent-a', 'agent-b']);
    expect(outcome.results).toEqual([
      { label: 'batch.json', status: 'success', message: '2 trajectories imported.' },
    ]);
  });

  it('reattributes ATIF to the pinned agent and counts the rewrites', async () => {
    const outcome = await ingestTraceFile(selected('trace.json', trajectory('from-the-file')), {
      workspace,
      agent: 'pinned',
    });

    expect(atifMock).toHaveBeenCalledWith(
      workspace,
      expect.objectContaining({
        agent: { name: 'pinned' },
      })
    );
    expect(outcome.agents).toEqual(['pinned']);
    expect(outcome.results[0].message).toContain('reattributed to "pinned"');
  });

  it('keeps importing after one trajectory fails', async () => {
    atifMock.mockRejectedValueOnce(new Error('boom')).mockResolvedValueOnce(undefined);

    const outcome = await ingestTraceFile(
      selected('batch.json', [trajectory('a'), trajectory('b')]),
      { workspace }
    );

    expect(atifMock).toHaveBeenCalledTimes(2);
    expect(outcome.results).toHaveLength(2);
    expect(outcome.results[0].status).toBe('success');
    expect(outcome.results[1]).toMatchObject({ status: 'error' });
  });

  it('batches direct spans under the 1000-span request cap', async () => {
    const spans = Array.from({ length: 1001 }, (_, index) => span(`span-${index}`));

    const outcome = await ingestTraceFile(selected('spans.json', spans), {
      workspace,
      source: 'langsmith',
    });

    expect(spansMock).toHaveBeenCalledTimes(2);
    expect(spansMock.mock.calls[0][1].spans).toHaveLength(1000);
    expect(spansMock.mock.calls[1][1].spans).toHaveLength(1);
    expect(outcome.results[0].message).toBe('1001 spans imported as source "langsmith".');
  });

  it('keeps the source a span batch names for itself', async () => {
    await ingestTraceFile(selected('spans.json', { source: 'mlflow', spans: [span('s1')] }), {
      workspace,
      source: 'ignored',
    });

    expect(spansMock.mock.calls[0][1].source).toBe('mlflow');
  });

  it('stamps the pinned agent onto every span, and reads it back', async () => {
    const outcome = await ingestTraceFile(
      selected('spans.json', [span('s1', { 'gen_ai.agent.name': 'from-the-file' })]),
      { workspace, agent: 'pinned' }
    );

    expect(spansMock.mock.calls[0][1].spans[0].attributes).toMatchObject({
      'gen_ai.agent.name': 'pinned',
    });
    expect(outcome.agents).toEqual(['pinned']);
  });

  it('posts one chat completion per call and claims no agent for them', async () => {
    const call = {
      request: { model: 'gpt-4o', messages: [{ role: 'user', content: 'hi' }] },
      response: { id: 'chatcmpl-1', model: 'gpt-4o', choices: [] },
    };

    const outcome = await ingestTraceFile(selected('calls.json', [call, call]), {
      workspace,
      agent: 'pinned',
    });

    expect(chatMock).toHaveBeenCalledTimes(2);
    expect(outcome.agents).toEqual([]);
    expect(outcome.results[0].message).toBe('2 model calls imported, not attributed to an agent.');
  });

  it('forwards an OTLP protobuf as raw bytes and surfaces its per-span errors', async () => {
    otlpMock.mockResolvedValueOnce({ errors: ['span 3 dropped'] });
    const file = new File(['binary'], 'export.binpb');
    const picked = await readTraceFile(file);

    const outcome = await ingestTraceFile(picked, { workspace });

    expect(otlpMock).toHaveBeenCalledWith(workspace, file);
    expect(outcome.results).toEqual([
      { label: 'export.binpb', status: 'error', message: 'span 3 dropped' },
    ]);
  });

  it('reports an undetected file without calling any endpoint', async () => {
    const outcome = await ingestTraceFile(selected('mystery.json', { hello: 'world' }), {
      workspace,
    });

    expect(outcome.results[0].status).toBe('error');
    expect(atifMock).not.toHaveBeenCalled();
    expect(spansMock).not.toHaveBeenCalled();
    expect(chatMock).not.toHaveBeenCalled();
    expect(otlpMock).not.toHaveBeenCalled();
  });

  it('caps a long tail of failures with a count of the rest', async () => {
    atifMock.mockRejectedValue(new Error('boom'));
    const batch = Array.from({ length: 9 }, (_, index) => trajectory(`agent-${index}`));

    const outcome = await ingestTraceFile(selected('batch.json', batch), { workspace });

    expect(outcome.results).toHaveLength(6);
    expect(outcome.results[5].message).toBe('...and 4 more failures.');
  });
});

describe('ingestTraceFiles', () => {
  it('routes a mixed selection to the endpoint each file belongs to', async () => {
    const outcome = await ingestTraceFiles(
      [
        selected('trace.json', trajectory('agent-a')),
        selected('spans.json', [span('s1', { 'agent.name': 'agent-b' })]),
        selected('mystery.json', { hello: 'world' }),
      ],
      { workspace }
    );

    expect(atifMock).toHaveBeenCalledTimes(1);
    expect(spansMock).toHaveBeenCalledTimes(1);
    expect(outcome.agents).toEqual(['agent-a', 'agent-b']);
    expect(outcome.results.map(({ status }) => status)).toEqual(['success', 'success', 'error']);
  });
});
