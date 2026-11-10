// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  agentFromPath,
  parseSpans,
  reduceSpansToRuns,
  runIdFromPath,
  type TelemetrySpan,
} from '@studio/routes/agents/AgentMonitorRoute/telemetry';

const SECONDS = 1_700_000_000;
const MS = 1_700_000_000_000;

describe('parseSpans', () => {
  it('parses one JSON object per non-empty line', () => {
    const text =
      '{"payload":{"event_type":"FUNCTION_END","UUID":"a"}}\n\n{"payload":{"event_type":"WORKFLOW_END","UUID":"b"}}\n';
    const spans = parseSpans(text);
    expect(spans).toHaveLength(2);
    expect(spans[0].payload.UUID).toBe('a');
  });

  it('skips malformed lines instead of throwing', () => {
    const text =
      '{"payload":{"event_type":"FUNCTION_END"}}\nnot-json\n{"payload":{"event_type":"FUNCTION_START"}}\n';
    expect(parseSpans(text)).toHaveLength(2);
  });

  it('augments parsed spans with agent and runId from the path', () => {
    const span = parseSpans('{"payload":{"event_type":"WORKFLOW_END"}}', 'agent-x', 'run-7')[0];
    expect(span.__agent).toBe('agent-x');
    expect(span.__runId).toBe('run-7');
  });

  it('skips non-object lines instead of throwing when tagging', () => {
    expect(
      parseSpans('null\n42\n{"payload":{"event_type":"WORKFLOW_END"}}', 'agent-x')
    ).toHaveLength(1);
  });
});

describe('agentFromPath / runIdFromPath', () => {
  it('extracts the agent segment', () => {
    expect(agentFromPath('agent-x/runs/run-1/part-0.jsonl')).toBe('agent-x');
    expect(agentFromPath('no-slash')).toBeUndefined();
  });

  it('extracts the run id only when the layout matches `<agent>/runs/<id>`', () => {
    expect(runIdFromPath('agent-x/runs/run-1/part-0.jsonl')).toBe('run-1');
    expect(runIdFromPath('agent-x/other/run-1/part.jsonl')).toBeUndefined();
    expect(runIdFromPath('flat.jsonl')).toBeUndefined();
  });
});

const span = (
  overrides: Partial<TelemetrySpan['payload']> & {
    __agent?: string;
    __runId?: string;
    parent_id?: string;
    function_ancestry?: TelemetrySpan['function_ancestry'];
  }
): TelemetrySpan => {
  const { __agent, __runId, parent_id, function_ancestry, ...payload } = overrides;
  const fullPayload: TelemetrySpan['payload'] = {
    event_type: 'FUNCTION_END',
    ...payload,
  };
  return { parent_id, function_ancestry, __agent, __runId, payload: fullPayload };
};

describe('reduceSpansToRuns', () => {
  it('groups spans by path-derived runId and aggregates tokens from streamed chunks', () => {
    const streamedOutput: unknown = [
      {
        choices: [{ delta: { content: 'done' } }],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      },
    ];
    const spans: TelemetrySpan[] = [
      span({
        __runId: 'r1',
        __agent: 'agent-a',
        event_type: 'WORKFLOW_START',
        event_timestamp: SECONDS,
        data: { input: { messages: [{ role: 'user', content: 'hello' }], model: 'mistral-7b' } },
      }),
      span({
        __runId: 'r1',
        event_type: 'FUNCTION_START',
        UUID: 'tool-1',
        event_timestamp: SECONDS + 1,
        function_ancestry: { function_name: 'clock', parent_name: '<workflow>' },
      }),
      span({
        __runId: 'r1',
        event_type: 'FUNCTION_END',
        UUID: 'tool-1', // duplicate UUID — counted once
        event_timestamp: SECONDS + 2,
        function_ancestry: { function_name: 'clock', parent_name: '<workflow>' },
      }),
      span({
        __runId: 'r1',
        event_type: 'FUNCTION_END',
        UUID: 'wf-fn',
        event_timestamp: SECONDS + 3,
        function_ancestry: { function_name: '<workflow>', parent_name: 'root' },
      }),
      span({
        __runId: 'r1',
        event_type: 'WORKFLOW_END',
        event_timestamp: SECONDS + 4,
        data: { output: streamedOutput },
      }),
    ];
    const [run] = reduceSpansToRuns(spans);
    expect(run.runId).toBe('r1');
    expect(run.agent).toBe('agent-a');
    expect(run.model).toBe('mistral-7b');
    expect(run.inputPreview).toBe('hello');
    expect(run.outputPreview).toBe('done');
    expect(run.promptTokens).toBe(10);
    expect(run.completionTokens).toBe(5);
    expect(run.toolCalls).toBe(1);
    expect(run.durationMs).toBe(4_000);
  });

  it('falls back to LLM_* span payload.name when WORKFLOW_START.input.model is null', () => {
    // Real NAT workflow runs leave input.model null because the agent doesn't
    // pass a model override down — langchain's LLM_* spans carry the deployed
    // model in payload.name.
    const spans: TelemetrySpan[] = [
      span({
        __runId: 'r',
        event_type: 'WORKFLOW_START',
        event_timestamp: SECONDS,
        data: {
          input: {
            messages: [{ role: 'user', content: 'hi' }],
            // model: null in real telemetry — represented as undefined here.
          },
        },
      }),
      span({
        __runId: 'r',
        event_type: 'LLM_START',
        event_timestamp: SECONDS + 1,
        name: 'nvidia-nemotron-3-nano-30b-a3b',
      }),
      span({ __runId: 'r', event_type: 'WORKFLOW_END', event_timestamp: SECONDS + 2 }),
    ];
    expect(reduceSpansToRuns(spans)[0].model).toBe('nvidia-nemotron-3-nano-30b-a3b');
  });

  it('treats `<workflow>` and `root` function spans as non-tools', () => {
    const spans: TelemetrySpan[] = [
      span({ __runId: 'r', event_type: 'WORKFLOW_START', event_timestamp: SECONDS }),
      span({
        __runId: 'r',
        event_type: 'FUNCTION_END',
        UUID: 'wf',
        event_timestamp: SECONDS + 1,
        function_ancestry: { function_name: '<workflow>' },
      }),
      span({
        __runId: 'r',
        event_type: 'FUNCTION_END',
        UUID: 'rt',
        event_timestamp: SECONDS + 2,
        function_ancestry: { function_name: 'root' },
      }),
      span({ __runId: 'r', event_type: 'WORKFLOW_END', event_timestamp: SECONDS + 3 }),
    ];
    expect(reduceSpansToRuns(spans)[0].toolCalls).toBe(0);
  });

  it('falls back to the legacy payload-level `usage_info.token_usage` shape', () => {
    const spans: TelemetrySpan[] = [
      span({ __runId: 'r', event_type: 'WORKFLOW_START', event_timestamp: SECONDS }),
      span({
        __runId: 'r',
        event_type: 'FUNCTION_END',
        UUID: 'leaf',
        event_timestamp: SECONDS + 1,
        usage_info: { token_usage: { prompt_tokens: 7, completion_tokens: 3 } },
      }),
      span({ __runId: 'r', event_type: 'WORKFLOW_END', event_timestamp: SECONDS + 2 }),
    ];
    const [run] = reduceSpansToRuns(spans);
    expect(run.promptTokens).toBe(7);
    expect(run.completionTokens).toBe(3);
  });

  it('prefers chunk-level usage over legacy usage_info to avoid double-counting', () => {
    // Mixed exporter run: one span carries legacy `usage_info`, another carries
    // streamed chunks with `usage`. Total should match the chunk numbers, not
    // the sum.
    const streamedOutput: unknown = [
      {
        choices: [{ delta: { content: 'ok' } }],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      },
    ];
    const spans: TelemetrySpan[] = [
      span({ __runId: 'r', event_type: 'WORKFLOW_START', event_timestamp: SECONDS }),
      span({
        __runId: 'r',
        event_type: 'FUNCTION_END',
        UUID: 'leaf',
        event_timestamp: SECONDS + 1,
        usage_info: { token_usage: { prompt_tokens: 99, completion_tokens: 99 } },
      }),
      span({
        __runId: 'r',
        event_type: 'WORKFLOW_END',
        event_timestamp: SECONDS + 2,
        data: { output: streamedOutput },
      }),
    ];
    const [run] = reduceSpansToRuns(spans);
    expect(run.promptTokens).toBe(10);
    expect(run.completionTokens).toBe(5);
  });

  it('reads Anthropic-style `input_tokens` / `output_tokens` aliases from chunk usage', () => {
    const streamedOutput: unknown = [
      { choices: [{ delta: { content: 'hi' } }], usage: { input_tokens: 4, output_tokens: 2 } },
    ];
    const spans: TelemetrySpan[] = [
      span({ __runId: 'r', event_type: 'WORKFLOW_START', event_timestamp: SECONDS }),
      span({
        __runId: 'r',
        event_type: 'WORKFLOW_END',
        event_timestamp: SECONDS + 1,
        data: { output: streamedOutput },
      }),
    ];
    const [run] = reduceSpansToRuns(spans);
    expect(run.promptTokens).toBe(4);
    expect(run.completionTokens).toBe(2);
  });

  it('extracts usage and assistant text from a non-streaming WORKFLOW_END output dict', () => {
    const nonStreamedOutput: unknown = {
      id: 'da8486b9-87cd-45f2-9b18-c1f2f5ff219a',
      object: 'chat.completion',
      model: 'unknown-model',
      choices: [
        { finish_reason: 'stop', index: 0, message: { role: 'assistant', content: '42.0' } },
      ],
      usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
    };
    const spans: TelemetrySpan[] = [
      span({ __runId: 'r', event_type: 'WORKFLOW_START', event_timestamp: SECONDS }),
      span({
        __runId: 'r',
        event_type: 'WORKFLOW_END',
        event_timestamp: SECONDS + 1,
        data: { output: nonStreamedOutput },
      }),
    ];
    const [run] = reduceSpansToRuns(spans);
    expect(run.promptTokens).toBe(5);
    expect(run.completionTokens).toBe(1);
    expect(run.outputPreview).toBe('42.0');
  });

  it('walks parent_id → UUID → workflow_run_id when path runId is missing', () => {
    const spans: TelemetrySpan[] = [
      span({
        UUID: 'wf',
        event_type: 'WORKFLOW_START',
        event_timestamp: SECONDS,
        metadata: { provided_metadata: { workflow_run_id: 'r-walked' } },
      }),
      span({
        UUID: 'mid',
        parent_id: 'wf',
        event_type: 'FUNCTION_START',
        event_timestamp: SECONDS,
        function_ancestry: { function_name: '<workflow>' },
      }),
      span({
        parent_id: 'mid',
        event_type: 'FUNCTION_END',
        UUID: 'tool',
        event_timestamp: SECONDS + 1,
        function_ancestry: { function_name: 'clock', parent_name: '<workflow>' },
      }),
    ];
    const [run] = reduceSpansToRuns(spans);
    expect(run.runId).toBe('r-walked');
    expect(run.toolCalls).toBe(1);
  });

  it('drops orphan spans that cannot be attributed to a run', () => {
    const spans: TelemetrySpan[] = [
      span({ event_type: 'FUNCTION_END', UUID: 'orphan', event_timestamp: SECONDS }),
    ];
    expect(reduceSpansToRuns(spans)).toEqual([]);
  });

  it('treats values > 1e12 as ms and smaller values as seconds', () => {
    const inSeconds: TelemetrySpan[] = [
      span({ __runId: 's', event_type: 'WORKFLOW_START', event_timestamp: SECONDS }),
      span({ __runId: 's', event_type: 'WORKFLOW_END', event_timestamp: SECONDS + 1 }),
    ];
    const inMs: TelemetrySpan[] = [
      span({ __runId: 'm', event_type: 'WORKFLOW_START', event_timestamp: MS }),
      span({ __runId: 'm', event_type: 'WORKFLOW_END', event_timestamp: MS + 1_000 }),
    ];
    const [secRun] = reduceSpansToRuns(inSeconds);
    const [msRun] = reduceSpansToRuns(inMs);
    expect(secRun.startedAt.getTime()).toBe(SECONDS * 1000);
    expect(msRun.startedAt.getTime()).toBe(MS);
  });

  it('returns runs sorted newest-first', () => {
    const spans: TelemetrySpan[] = [
      span({ __runId: 'old', event_type: 'WORKFLOW_START', event_timestamp: SECONDS }),
      span({ __runId: 'old', event_type: 'WORKFLOW_END', event_timestamp: SECONDS + 1 }),
      span({ __runId: 'new', event_type: 'WORKFLOW_START', event_timestamp: SECONDS + 100 }),
      span({ __runId: 'new', event_type: 'WORKFLOW_END', event_timestamp: SECONDS + 101 }),
    ];
    const runs = reduceSpansToRuns(spans);
    expect(runs.map((r) => r.runId)).toEqual(['new', 'old']);
  });

  it('extracts streamed delta content for outputPreview', () => {
    const streamed: unknown = [
      { choices: [{ delta: { content: 'Hel' } }] },
      { choices: [{ delta: { content: 'lo!' } }] },
    ];
    const spans: TelemetrySpan[] = [
      span({ __runId: 's', event_type: 'WORKFLOW_START', event_timestamp: SECONDS }),
      span({
        __runId: 's',
        event_type: 'WORKFLOW_END',
        event_timestamp: SECONDS + 1,
        data: { output: streamed },
      }),
    ];
    const [run] = reduceSpansToRuns(spans);
    expect(run.outputPreview).toBe('Hello!');
  });
});
