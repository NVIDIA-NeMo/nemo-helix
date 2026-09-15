// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type AtifIngestRequest,
  AtifIngestRequestSchemaVersion,
} from '@nemo/sdk/generated/platform/schema';

const SCHEMA_VERSIONS = new Set<string>(Object.values(AtifIngestRequestSchemaVersion));

export interface ParsedTrace {
  /** Where this trajectory came from — a file name, or "Pasted JSON" plus an index. */
  label: string;
  trajectory: AtifIngestRequest;
}

export interface ParseFailure {
  label: string;
  message: string;
}

export interface ParseAtifResult {
  traces: ParsedTrace[];
  failures: ParseFailure[];
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * Front-loads the checks Intake would otherwise reject with a 422, so a typo in a
 * hand-written trace surfaces next to its file name instead of as one opaque failure.
 * This is deliberately shallow — the service remains the authority on the full schema.
 */
const validate = (value: unknown): string | null => {
  if (!isRecord(value)) return 'Expected a JSON object describing one ATIF trajectory.';

  const { schema_version: schemaVersion, agent, steps } = value;

  if (typeof schemaVersion !== 'string') return 'Missing required field "schema_version".';
  if (!SCHEMA_VERSIONS.has(schemaVersion)) {
    return `Unsupported schema_version "${schemaVersion}". Expected one of ${[...SCHEMA_VERSIONS].join(', ')}.`;
  }

  if (!isRecord(agent)) return 'Missing required field "agent".';
  if (typeof agent.name !== 'string' || agent.name.length === 0) {
    return 'Field "agent.name" must be a non-empty string.';
  }

  if (steps !== undefined) {
    if (!Array.isArray(steps)) return 'Field "steps" must be an array.';
    const misnumbered = steps.findIndex(
      (step, index) => !isRecord(step) || step.step_id !== index + 1
    );
    if (misnumbered !== -1) {
      return `Step ${misnumbered + 1} must be an object with "step_id": ${misnumbered + 1} — step IDs are 1-based and sequential.`;
    }
  }

  return null;
};

/**
 * Validates one already-parsed JSON body into ATIF trajectories. A body may hold a single
 * trajectory or an array of them, so a whole exported batch can be imported at once.
 */
export const parseAtifValue = (label: string, parsed: unknown): ParseAtifResult => {
  const candidates = Array.isArray(parsed) ? parsed : [parsed];
  if (candidates.length === 0) {
    return { traces: [], failures: [{ label, message: 'The array is empty.' }] };
  }

  const traces: ParsedTrace[] = [];
  const failures: ParseFailure[] = [];

  candidates.forEach((candidate, index) => {
    // Only disambiguate by position when the document actually held several.
    const itemLabel = candidates.length > 1 ? `${label} [${index + 1}]` : label;
    const message = validate(candidate);
    if (message) {
      failures.push({ label: itemLabel, message });
    } else {
      traces.push({ label: itemLabel, trajectory: candidate as AtifIngestRequest });
    }
  });

  return { traces, failures };
};

/** Parses one JSON document into ATIF trajectories. */
export const parseAtifDocument = (label: string, text: string): ParseAtifResult => {
  if (text.trim().length === 0) return { traces: [], failures: [] };

  try {
    return parseAtifValue(label, JSON.parse(text));
  } catch (error) {
    return {
      traces: [],
      failures: [{ label, message: error instanceof Error ? error.message : 'Invalid JSON.' }],
    };
  }
};

export const parseAtifDocuments = (documents: { label: string; text: string }[]): ParseAtifResult =>
  documents.reduce<ParseAtifResult>(
    (accumulated, { label, text }) => {
      const { traces, failures } = parseAtifDocument(label, text);
      return {
        traces: [...accumulated.traces, ...traces],
        failures: [...accumulated.failures, ...failures],
      };
    },
    { traces: [], failures: [] }
  );

/**
 * Reattributes a trajectory to `agent`.
 *
 * Intake derives a span's `agent_name` from ATIF's `agent.name`, so importing from an
 * agent's own page has to rewrite that field — otherwise the traces land in Intake
 * unattached to the agent whose page they were imported from. The rest of the agent
 * block (version, model_name) is preserved.
 */
export const applyAgentName = (
  trajectory: AtifIngestRequest,
  agent: string
): AtifIngestRequest => ({
  ...trajectory,
  agent: { ...trajectory.agent, name: agent },
});

/** The agent a trajectory names, when it differs from `agent`. */
export const reattributedFrom = (
  trajectory: AtifIngestRequest,
  agent: string
): string | undefined => {
  const original = trajectory.agent?.name;
  return original && original !== agent ? original : undefined;
};
