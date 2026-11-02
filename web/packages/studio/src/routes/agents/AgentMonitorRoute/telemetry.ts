// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseJsonl } from '@nemo/common/src/utils/parseRecords';

export interface TelemetrySpan {
  parent_id?: string;
  function_ancestry?: {
    function_id?: string;
    function_name?: string;
    parent_id?: string | null;
    parent_name?: string | null;
  };
  payload: {
    UUID?: string;
    event_type: string;
    event_timestamp?: number;
    name?: string;
    metadata?: {
      provided_metadata?: {
        workflow_run_id?: string;
        workflow_trace_id?: string;
        conversation_id?: string;
        display_name?: string;
      };
    };
    data?: {
      // WORKFLOW_START: { messages, model }; WORKFLOW_END: streamed chat-completion chunks.
      input?: string | { messages?: { role: string; content?: string }[]; model?: string };
      output?: unknown;
    };
    usage_info?: {
      token_usage?: {
        prompt_tokens?: number;
        completion_tokens?: number;
        total_tokens?: number;
      };
    } | null;
  };
  /** Augmented from the file path (`<agent>/runs/...`). */
  __agent?: string;
  /** Augmented from the file path's `<run-id>` segment. */
  __runId?: string;
}

export interface RunSummary {
  runId: string;
  agent?: string;
  startedAt: Date;
  endedAt: Date;
  durationMs: number;
  model?: string;
  inputPreview: string;
  outputPreview: string;
  promptTokens: number;
  completionTokens: number;
  toolCalls: number;
}

const PREVIEW_MAX = 160;
// `function_name` values reserved by the exporter for the trace root and the
// agent workflow itself — everything else is a user-defined tool.
const NON_TOOL_FUNCTIONS = new Set(['root', '<workflow>']);

/** Parses a spans JSONL body, skipping malformed lines and tagging each span with its source. */
export const parseSpans = (text: string, agent?: string, runId?: string): TelemetrySpan[] =>
  parseJsonl<TelemetrySpan>(text, { skipInvalid: true }).map((span) => {
    if (agent) span.__agent = agent;
    if (runId) span.__runId = runId;
    return span;
  });

export const agentFromPath = (path: string): string | undefined => {
  const slash = path.indexOf('/');
  return slash > 0 ? path.slice(0, slash) : undefined;
};

// Path: `<agent>/runs/<run-id>/<part>.jsonl`
export const runIdFromPath = (path: string): string | undefined => {
  const parts = path.split('/');
  if (parts.length >= 3 && parts[1] === 'runs') return parts[2];
  return undefined;
};

const truncate = (text: string): string =>
  text.length > PREVIEW_MAX ? `${text.slice(0, PREVIEW_MAX)}…` : text;

const getRunId = (span: TelemetrySpan): string | undefined =>
  span.payload.metadata?.provided_metadata?.workflow_run_id;

const asChunkList = (output: unknown): unknown[] => {
  if (Array.isArray(output)) return output;
  if (typeof output === 'object' && output !== null) return [output];
  return [];
};

const choiceText = (choice: unknown): string => {
  if (typeof choice !== 'object' || choice === null) return '';
  const c = choice as Record<string, unknown>;
  const delta = c['delta'];
  if (typeof delta === 'object' && delta !== null) {
    const content = (delta as Record<string, unknown>)['content'];
    if (typeof content === 'string') return content;
  }
  const message = c['message'];
  if (typeof message === 'object' && message !== null) {
    const content = (message as Record<string, unknown>)['content'];
    if (typeof content === 'string') return content;
  }
  return '';
};

const extractOutputText = (output: unknown): string => {
  if (typeof output === 'string') return output;
  return asChunkList(output)
    .flatMap((chunk) => {
      if (typeof chunk !== 'object' || chunk === null) return [];
      const choices = (chunk as Record<string, unknown>)['choices'];
      if (!Array.isArray(choices)) return [];
      return choices.map(choiceText).filter((text) => text.length > 0);
    })
    .join('');
};

interface UsageAcc {
  prompt: number;
  completion: number;
}

// Walks OpenAI-style streaming chunks and sums any per-chunk `usage` totals.
// Tolerates both `prompt_tokens`/`completion_tokens` and Anthropic-style
// `input_tokens`/`output_tokens` aliases.
const collectChunkUsage = (output: unknown, acc: UsageAcc): void => {
  for (const chunk of asChunkList(output)) {
    if (typeof chunk !== 'object' || chunk === null) continue;
    const usage = (chunk as Record<string, unknown>)['usage'];
    if (typeof usage !== 'object' || usage === null) continue;
    const u = usage as Record<string, unknown>;
    const prompt = u['prompt_tokens'] ?? u['input_tokens'];
    const completion = u['completion_tokens'] ?? u['output_tokens'];
    if (typeof prompt === 'number') acc.prompt += prompt;
    if (typeof completion === 'number') acc.completion += completion;
  }
};

export const reduceSpansToRuns = (spans: TelemetrySpan[]): RunSummary[] => {
  // Only WORKFLOW_* spans carry workflow_run_id; others must walk parent_id →
  // payload.UUID.
  const uuidToRunId = new Map<string, string>();
  const uuidToParentId = new Map<string, string>();
  for (const span of spans) {
    const uuid = span.payload.UUID;
    if (!uuid) continue;
    const runId = getRunId(span);
    if (runId) uuidToRunId.set(uuid, runId);
    if (span.parent_id && span.parent_id !== 'root') uuidToParentId.set(uuid, span.parent_id);
  }

  const resolveRunId = (span: TelemetrySpan): string | undefined => {
    const direct = getRunId(span);
    if (direct) return direct;
    let currentId = span.parent_id;
    for (let i = 0; i < 8 && currentId && currentId !== 'root'; i++) {
      const found = uuidToRunId.get(currentId);
      if (found) return found;
      currentId = uuidToParentId.get(currentId);
    }
    return undefined;
  };

  const grouped = new Map<string, TelemetrySpan[]>();
  for (const span of spans) {
    // Path-derived run id is most reliable — exporter writes one run per dir.
    // Fall back to workflow_run_id chain when file path wasn't propagated.
    const runId = span.__runId ?? resolveRunId(span);
    if (!runId) continue;
    const list = grouped.get(runId) ?? [];
    list.push(span);
    grouped.set(runId, list);
  }

  const runs: RunSummary[] = [];
  for (const [runId, runSpans] of grouped) {
    let startTs = Infinity;
    let endTs = -Infinity;
    const legacyUsage: UsageAcc = { prompt: 0, completion: 0 };
    const chunkUsage: UsageAcc = { prompt: 0, completion: 0 };
    // Dedupe by UUID across START/END so each tool invocation counts once.
    const toolCallUuids = new Set<string>();
    let model: string | undefined;
    let agent: string | undefined;
    let inputPreview = '';
    let outputPreview = '';

    for (const span of runSpans) {
      const p = span.payload;
      if (typeof p.event_timestamp === 'number') {
        if (p.event_timestamp < startTs) startTs = p.event_timestamp;
        if (p.event_timestamp > endTs) endTs = p.event_timestamp;
      }
      if (!agent && span.__agent) agent = span.__agent;

      // langchain LLM_* spans put the deployed model name in payload.name. The
      // workflow's input.model is often null because the agent doesn't pass a
      // model override down — use the LLM span as the source of truth.
      if (!model && p.event_type.startsWith('LLM_') && typeof p.name === 'string' && p.name) {
        model = p.name;
      }

      // Older NAT exporters surfaced usage at payload-level — keep harvesting
      // it so a partial schema rollback doesn't blank out tokens.
      const tokenUsage = p.usage_info?.token_usage;
      if (tokenUsage) {
        if (typeof tokenUsage.prompt_tokens === 'number')
          legacyUsage.prompt += tokenUsage.prompt_tokens;
        if (typeof tokenUsage.completion_tokens === 'number')
          legacyUsage.completion += tokenUsage.completion_tokens;
      }

      switch (p.event_type) {
        case 'WORKFLOW_START': {
          const input = p.data?.input;
          if (typeof input === 'string') {
            if (!inputPreview) inputPreview = truncate(input);
          } else if (input) {
            const msgs = input.messages ?? [];
            for (let i = msgs.length - 1; i >= 0; i -= 1) {
              const m = msgs[i];
              if (m.role === 'user' && typeof m.content === 'string') {
                inputPreview = truncate(m.content);
                break;
              }
            }
            if (!model && input.model) model = input.model;
          }
          break;
        }
        case 'WORKFLOW_END': {
          const text = extractOutputText(p.data?.output);
          if (text.length > 0) outputPreview = truncate(text);
          // Streamed chunks carry per-chunk `usage` when the provider sets it.
          collectChunkUsage(p.data?.output, chunkUsage);
          break;
        }
        case 'FUNCTION_START':
        case 'FUNCTION_END': {
          // The agent itself runs as a `<workflow>` FUNCTION span; nested
          // functions are the user-defined tools.
          const fnName = span.function_ancestry?.function_name;
          if (fnName && !NON_TOOL_FUNCTIONS.has(fnName) && p.UUID) {
            toolCallUuids.add(p.UUID);
          }
          break;
        }
        default:
          break;
      }
    }

    if (!Number.isFinite(startTs) || !Number.isFinite(endTs)) continue;

    const hasChunkUsage = chunkUsage.prompt > 0 || chunkUsage.completion > 0;
    const finalUsage = hasChunkUsage ? chunkUsage : legacyUsage;

    // NAT emits seconds-since-epoch as float; detect ms defensively.
    const toMs = (ts: number) => (ts > 1e12 ? ts : ts * 1000);
    runs.push({
      runId,
      agent,
      startedAt: new Date(toMs(startTs)),
      endedAt: new Date(toMs(endTs)),
      durationMs: Math.max(0, toMs(endTs) - toMs(startTs)),
      model,
      inputPreview,
      outputPreview,
      promptTokens: finalUsage.prompt,
      completionTokens: finalUsage.completion,
      toolCalls: toolCallUuids.size,
    });
  }

  return runs.sort((a, b) => b.startedAt.getTime() - a.startedAt.getTime());
};
