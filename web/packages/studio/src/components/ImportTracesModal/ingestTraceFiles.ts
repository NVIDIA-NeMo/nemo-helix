// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import {
  ingestAtif,
  ingestChatCompletion,
  ingestOtlpTraces,
  ingestSpans,
} from '@nemo/sdk/generated/platform/ingest';
import type {
  ChatCompletionsIngestRequest,
  DirectSpanInput,
} from '@nemo/sdk/generated/platform/schema';
import {
  type Detection,
  detectTraceFormat,
  isProtobufFileName,
} from '@studio/components/ImportTracesModal/detectTraceFormat';
import {
  applyAgentName,
  parseAtifValue,
  reattributedFrom,
} from '@studio/components/ImportTracesModal/parseAtifTraces';
import type { ImportTraceResult } from '@studio/components/ImportTracesModal/types';

/** Source name recorded for spans uploaded here rather than pulled by an importer. */
export const DEFAULT_SPANS_SOURCE = 'studio-upload';

/** Intake rejects a direct-span request carrying more than this many spans. */
const SPANS_PER_REQUEST = 1000;

/** How many per-record failures one file reports before the rest are counted instead. */
const MAX_ERROR_ROWS = 5;

/** The attribute Intake reads a span's `agent_name` from, first alias wins. */
const AGENT_NAME_KEYS = ['gen_ai.agent.name', 'llm.agent.name', 'agent.name'];

export interface SelectedTraceFile {
  /** Name, size, and mtime — enough to recognize the same file picked twice. */
  id: string;
  label: string;
  file: File;
  detection: Detection;
}

export interface IngestOptions {
  workspace: string;
  /** Attribute every imported record to this agent, where the format allows it. */
  agent?: string;
  /** `source` recorded on direct-span batches. */
  source?: string;
}

export interface IngestOutcome {
  results: ImportTraceResult[];
  /** Agents named by records that imported successfully. Drives the insights trigger. */
  agents: string[];
}

/** Reads a picked file and names the ingest endpoint it belongs to. */
export const readTraceFile = async (file: File): Promise<SelectedTraceFile> => {
  const text = isProtobufFileName(file.name) ? null : await file.text();
  return {
    id: `${file.name}:${file.size}:${file.lastModified}`,
    label: file.name,
    file,
    detection: detectTraceFormat(file.name, text),
  };
};

const messageOf = (error: unknown): string =>
  error instanceof Error ? getErrorMessage(error) || error.message : 'Import failed.';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const plural = (count: number, noun: string) => `${count} ${noun}${count === 1 ? '' : 's'}`;

/** Keeps a long tail of per-record failures from burying the summary. */
const cap = (label: string, errors: string[]): ImportTraceResult[] => {
  const shown = errors.slice(0, MAX_ERROR_ROWS).map((message) => ({
    label,
    status: 'error' as const,
    message,
  }));
  const hidden = errors.length - shown.length;
  return hidden > 0
    ? [...shown, { label, status: 'error', message: `...and ${plural(hidden, 'more failure')}.` }]
    : shown;
};

const chunk = <T>(items: T[], size: number): T[][] => {
  const batches: T[][] = [];
  for (let index = 0; index < items.length; index += size) {
    batches.push(items.slice(index, index + size));
  }
  return batches;
};

const spanAgentName = (span: DirectSpanInput): string | undefined => {
  const attributes = span.attributes ?? {};
  for (const key of AGENT_NAME_KEYS) {
    const value = attributes[key];
    if (typeof value === 'string' && value.length > 0) return value;
  }
  return undefined;
};

const ingestAtifFile = async (
  { label, detection }: SelectedTraceFile,
  { workspace, agent }: IngestOptions
): Promise<IngestOutcome> => {
  const { traces, failures } = parseAtifValue(
    label,
    'document' in detection ? detection.document : undefined
  );
  const errors = failures.map(({ label: item, message }) =>
    item === label ? message : `${item}: ${message}`
  );
  const agents: string[] = [];
  let imported = 0;
  let reattributed = 0;

  for (const { label: item, trajectory } of traces) {
    const data = agent ? applyAgentName(trajectory, agent) : trajectory;
    try {
      await ingestAtif(workspace, data);
      imported += 1;
      if (agent && reattributedFrom(trajectory, agent)) reattributed += 1;
      const name = data.agent?.name;
      if (name) agents.push(name);
    } catch (error) {
      errors.push(`${item}: ${messageOf(error)}`);
    }
  }

  const results = cap(label, errors);
  if (imported > 0) {
    results.unshift({
      label,
      status: 'success',
      message: `${imported} ${imported === 1 ? 'trajectory' : 'trajectories'} imported${
        reattributed > 0 ? `, ${reattributed} reattributed to "${agent}"` : ''
      }.`,
    });
  }
  return { results, agents };
};

const ingestSpansFile = async (
  { label, detection }: SelectedTraceFile,
  { workspace, agent, source }: IngestOptions
): Promise<IngestOutcome> => {
  const document = 'document' in detection ? detection.document : undefined;
  const body = isRecord(document) ? document : { spans: document };
  const spans = Array.isArray(body.spans) ? (body.spans as DirectSpanInput[]) : [];
  const batchSource =
    (typeof body.source === 'string' && body.source) || source || DEFAULT_SPANS_SOURCE;

  if (spans.length === 0) return { results: cap(label, ['No spans found.']), agents: [] };

  const attributed = agent
    ? spans.map((span) => ({
        ...span,
        attributes: { ...span.attributes, 'gen_ai.agent.name': agent },
      }))
    : spans;

  const errors: string[] = [];
  const agents = new Set<string>();
  let imported = 0;

  const batches = chunk(attributed, SPANS_PER_REQUEST);
  for (const [index, batch] of batches.entries()) {
    try {
      await ingestSpans(workspace, { source: batchSource, spans: batch });
      imported += batch.length;
      for (const span of batch) {
        const name = spanAgentName(span);
        if (name) agents.add(name);
      }
    } catch (error) {
      errors.push(
        batches.length > 1
          ? `Batch ${index + 1} of ${batches.length}: ${messageOf(error)}`
          : messageOf(error)
      );
    }
  }

  const results = cap(label, errors);
  if (imported > 0) {
    results.unshift({
      label,
      status: 'success',
      message: `${plural(imported, 'span')} imported as source "${batchSource}".`,
    });
  }
  return { results, agents: [...agents] };
};

const ingestChatCompletionsFile = async (
  { label, detection }: SelectedTraceFile,
  { workspace }: IngestOptions
): Promise<IngestOutcome> => {
  const document = 'document' in detection ? detection.document : undefined;
  const calls = (Array.isArray(document) ? document : [document]) as ChatCompletionsIngestRequest[];

  const errors: string[] = [];
  let imported = 0;

  for (const [index, call] of calls.entries()) {
    try {
      await ingestChatCompletion(workspace, call);
      imported += 1;
    } catch (error) {
      errors.push(calls.length > 1 ? `Call ${index + 1}: ${messageOf(error)}` : messageOf(error));
    }
  }

  const results = cap(label, errors);
  if (imported > 0) {
    results.unshift({
      label,
      status: 'success',
      // Chat-completions ingest carries no agent field, so these spans are queryable
      // telemetry but never attach to an agent.
      message: `${plural(imported, 'model call')} imported, not attributed to an agent.`,
    });
  }
  return { results, agents: [] };
};

const ingestOtlpFile = async (
  { label, file }: SelectedTraceFile,
  { workspace }: IngestOptions
): Promise<IngestOutcome> => {
  try {
    const response = await ingestOtlpTraces(workspace, file);
    const errors = response?.errors ?? [];
    return {
      results:
        errors.length > 0
          ? cap(label, errors)
          : [
              {
                label,
                status: 'success',
                message: 'OTLP protobuf imported; the agent name comes from its own spans.',
              },
            ],
      agents: [],
    };
  } catch (error) {
    return { results: cap(label, [messageOf(error)]), agents: [] };
  }
};

/** Sends one picked file to the endpoint its detected format belongs to. */
export const ingestTraceFile = async (
  selected: SelectedTraceFile,
  options: IngestOptions
): Promise<IngestOutcome> => {
  const { detection, label } = selected;
  switch (detection.format) {
    case 'atif':
      return ingestAtifFile(selected, options);
    case 'spans':
      return ingestSpansFile(selected, options);
    case 'chat-completions':
      return ingestChatCompletionsFile(selected, options);
    case 'otlp-protobuf':
      return ingestOtlpFile(selected, options);
    default:
      return { results: [{ label, status: 'error', message: detection.message }], agents: [] };
  }
};

/**
 * Imports every picked file, one after another so the results read in the order they were
 * chosen. A file that fails does not stop the ones behind it.
 */
export const ingestTraceFiles = async (
  files: SelectedTraceFile[],
  options: IngestOptions
): Promise<IngestOutcome> => {
  const results: ImportTraceResult[] = [];
  const agents = new Set<string>();

  for (const file of files) {
    const outcome = await ingestTraceFile(file, options);
    results.push(...outcome.results);
    outcome.agents.forEach((agent) => agents.add(agent));
  }

  return { results, agents: [...agents] };
};
