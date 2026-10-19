#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable no-console -- CLI script */
/**
 * Re-expresses the sample ATIF trajectories in the other formats Intake ingests.
 *
 * Studio's import modal routes each picked file by sniffing its shape, so exercising that
 * routing needs one real file per shape. This converts a few of the ATIF traces beside it into
 * direct spans and captured chat completions, without inventing new content — the converted
 * files describe the same runs as their sources.
 *
 * The two `formats/otlp-traces.*` fixtures are deliberately not regenerated here: encoding OTLP
 * needs a protobuf writer, and neither Studio nor this script has one. They are checked in as
 * static fixtures — see the README beside them for their provenance and how to rebuild them.
 *
 *   pnpm traces:convert
 */
import type {
  AtifIngestRequest,
  AtifStep,
  AtifStepAgent,
  ChatCompletionsIngestRequest,
  DirectSpanInput,
} from '@nemo/sdk/generated/platform/schema';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const TRACES_DIR = resolve(__dirname, '../src/mocks/email-security-triage-traces');

/** How long a step's LLM span is taken to run. The traces only carry a start per step. */
const STEP_SECONDS = 6;

const SERVICE_NAME = 'email-security-triage';

type SpanAttributes = Record<string, string | number | boolean>;

/** A converted span. Everything the ATIF sources can fill in is present. */
interface ConvertedSpan extends DirectSpanInput {
  session_id: string;
  name: string;
  started_at: string;
  ended_at: string;
  input: string | null;
  output: string | null;
  attributes: SpanAttributes;
}

const toIso = (moment: Date): string => `${moment.toISOString().slice(0, 19)}Z`;

const plus = (stamp: string, seconds: number): Date => new Date(Date.parse(stamp) + seconds * 1000);

/** A stable hex identifier of `width` characters. Stable across runs, unlike a random id. */
const hexId = (seed: string, width: number): string =>
  createHash('sha256').update(seed).digest('hex').slice(0, width);

const isAgentStep = (step: AtifStep): step is AtifStepAgent => step.source === 'agent';

/** Pulls the configured timeout out of an `AgentTimeoutError` message, e.g. "... after 120.0 seconds". */
const timeoutSeconds = (error: { type?: string; message?: string } | undefined): number | null => {
  if (error?.type !== 'AgentTimeoutError' || !error.message) return null;
  const match = /after ([\d.]+) seconds/.exec(error.message);
  return match ? Number(match[1]) : null;
};

/** ATIF messages may be content parts; the sample set only ever uses plain strings. */
const messageText = (step: AtifStep): string =>
  typeof step.message === 'string' ? step.message : '';

const defined = (attributes: Record<string, unknown>): SpanAttributes =>
  Object.fromEntries(
    Object.entries(attributes).filter(([, value]) => value !== undefined && value !== null)
  ) as SpanAttributes;

/** One AGENT root span for the run, plus one LLM span per agent step. */
const toSpans = (trace: AtifIngestRequest): ConvertedSpan[] => {
  const session = trace.session_id ?? '';
  const traceId = hexId(`trace:${session}`, 32);
  const rootId = hexId(`root:${session}`, 16);
  const { agent } = trace;
  const error = trace.extra?.error as { type?: string; message?: string } | undefined;
  const steps = trace.steps ?? [];

  const userStep = steps.find((step) => step.source === 'user');
  const agentSteps = steps.filter(isAgentStep);
  const prompt = userStep ? messageText(userStep) : null;

  const started = new Date(Date.parse(steps[0].timestamp ?? ''));
  const timeout = agentSteps.length === 0 ? timeoutSeconds(error) : null;
  const ended =
    timeout !== null
      ? plus(steps[0].timestamp ?? '', timeout)
      : plus(steps[steps.length - 1].timestamp ?? '', STEP_SECONDS);

  const spans: ConvertedSpan[] = [
    {
      span_id: rootId,
      trace_id: traceId,
      session_id: session,
      name: `${SERVICE_NAME}.run`,
      kind: 'AGENT',
      status: error ? 'error' : 'success',
      started_at: toIso(started),
      ended_at: toIso(ended),
      input: prompt,
      output: agentSteps.length > 0 ? messageText(agentSteps[agentSteps.length - 1]) : null,
      attributes: defined({
        'gen_ai.agent.name': agent.name,
        'gen_ai.agent.version': agent.version,
        'gen_ai.request.model': agent.model_name,
        'gen_ai.usage.input_tokens': trace.final_metrics?.total_prompt_tokens,
        'gen_ai.usage.output_tokens': trace.final_metrics?.total_completion_tokens,
        ...(error ? { 'error.type': error.type, 'error.message': error.message } : {}),
      }),
    },
  ];

  for (const step of agentSteps) {
    const metrics = step.metrics ?? {};
    spans.push({
      span_id: hexId(`step:${session}:${step.step_id}`, 16),
      trace_id: traceId,
      session_id: session,
      parent_span_id: rootId,
      name: 'chat.completions',
      kind: 'LLM',
      status: 'success',
      started_at: toIso(new Date(Date.parse(step.timestamp ?? ''))),
      ended_at: toIso(plus(step.timestamp ?? '', STEP_SECONDS)),
      input: prompt,
      output: messageText(step),
      attributes: defined({
        'gen_ai.agent.name': agent.name,
        'gen_ai.request.model': step.model_name ?? agent.model_name,
        'gen_ai.usage.input_tokens': metrics.prompt_tokens,
        'gen_ai.usage.output_tokens': metrics.completion_tokens,
        'llm.cost.total': metrics.cost_usd,
      }),
    });
  }

  return spans;
};

/** One captured request/response pair per agent step. */
const toChatCompletions = (trace: AtifIngestRequest): ChatCompletionsIngestRequest[] => {
  const session = trace.session_id ?? '';
  const steps = trace.steps ?? [];
  const userStep = steps.find((step) => step.source === 'user');
  const prompt = userStep ? messageText(userStep) : '';

  return steps.filter(isAgentStep).map((step) => {
    const metrics = step.metrics ?? {};
    const model = step.model_name ?? trace.agent.model_name ?? '';
    const promptTokens = metrics.prompt_tokens ?? 0;
    const completionTokens = metrics.completion_tokens ?? 0;

    return {
      session_id: session,
      provider: 'nvidia',
      cost_usd: metrics.cost_usd,
      request: {
        model,
        messages: [{ role: 'user', content: prompt }],
      },
      response: {
        id: `chatcmpl-${hexId(`${session}:${step.step_id}`, 12)}`,
        object: 'chat.completion',
        created: Math.floor(Date.parse(step.timestamp ?? '') / 1000),
        model,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: messageText(step) },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: promptTokens,
          completion_tokens: completionTokens,
          total_tokens: promptTokens + completionTokens,
        },
      },
    };
  });
};

const writeJson = async (path: string, payload: unknown) => {
  await writeFile(path, `${JSON.stringify(payload, null, 2)}\n`);
  console.log(`wrote ${path.slice(TRACES_DIR.length + 1)}`);
};

const main = async () => {
  const out = join(TRACES_DIR, 'formats');
  await mkdir(out, { recursive: true });

  const load = async (name: string): Promise<AtifIngestRequest> =>
    JSON.parse(await readFile(join(TRACES_DIR, name), 'utf8')) as AtifIngestRequest;

  const correct = await load('trace-17-triage-correct.json');
  const benign = await load('trace-02-triage-benign-invoice.json');
  const timeout = await load('trace-06-timeout-error.json');
  const misroute = await load('trace-07-misroute-multiselect.json');

  // A bare array of spans: what a hand-rolled exporter produces.
  await writeJson(join(out, 'spans-array.json'), toSpans(correct));

  // A wrapped batch naming its own source, the way the provider importers post.
  await writeJson(join(out, 'spans-batch.json'), {
    source: 'langsmith',
    spans: [...toSpans(benign), ...toSpans(timeout)],
  });

  // Line-delimited spans, which is how most trace stores stream an export.
  const lines = toSpans(misroute).map((span) => JSON.stringify(span));
  await writeFile(join(out, 'spans.jsonl'), `${lines.join('\n')}\n`);
  console.log('wrote formats/spans.jsonl');

  await writeJson(join(out, 'chat-completions.json'), [
    ...toChatCompletions(correct),
    ...toChatCompletions(benign),
  ]);
};

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack : error);
  process.exitCode = 1;
});
