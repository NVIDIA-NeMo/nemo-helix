// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isPlainObject } from '@nemo/common/src/utils/isPlainObject';
import type { Rails } from '@nemo/sdk/generated/platform/schema';
import { recognizeFlow } from '@studio/routes/guardrails/GuardrailConfigTab/flowRegistry';
import type { DetectorKey, Field, Scope } from '@studio/routes/guardrails/GuardrailConfigTab/types';

export interface DetectorMeta {
  label: string;
  /** True for NVIDIA / first-party detectors, false for third-party integrations. */
  firstParty: boolean;
}

/**
 * Display metadata and canonical ordering for the known `rails.config.*`
 * providers. Detectors are listed first-party first, then third-party.
 */
export const DETECTOR_META: Record<string, DetectorMeta> = {
  content_safety: { label: 'Content Safety', firstParty: true },
  jailbreak_detection: { label: 'Jailbreak Detection', firstParty: true },
  sensitive_data_detection: { label: 'Sensitive Data (Presidio)', firstParty: true },
  gliner: { label: 'PII — GLiNER', firstParty: true },
  privateai: { label: 'PII — Private AI', firstParty: true },
  polygraf: { label: 'PII — Polygraf', firstParty: true },
  regex_detection: { label: 'Regex Detection', firstParty: true },
  injection_detection: { label: 'Injection Detection', firstParty: true },
  context_bloat_detection: { label: 'Context Bloat Detection', firstParty: true },
  fact_checking: { label: 'Fact Checking', firstParty: true },
  hf_classifier: { label: 'HuggingFace Classifiers', firstParty: true },
  autoalign: { label: 'AutoAlign', firstParty: false },
  patronus: { label: 'Patronus', firstParty: false },
  clavata: { label: 'Clavata', firstParty: false },
  pangea: { label: 'Pangea AI Guard', firstParty: false },
  fiddler: { label: 'Fiddler Guardrails', firstParty: false },
  crowdstrike_aidr: { label: 'CrowdStrike AIDR', firstParty: false },
  trend_micro: { label: 'Trend Micro AI Guard', firstParty: false },
  ai_defense: { label: 'Cisco AI Defense', firstParty: false },
  guardrails_ai: { label: 'Guardrails AI', firstParty: false },
};

const DETECTOR_ORDER = Object.keys(DETECTOR_META);

/** Turn a snake_case key into a human-readable Title Case label. */
export const humanize = (key: string): string =>
  key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .trim();

/** Resolve display metadata for a detector key, falling back for unknown keys. */
export const detectorMeta = (key: string): DetectorMeta =>
  DETECTOR_META[key] ?? { label: humanize(key), firstParty: false };

/**
 * Enumerate the detectors actually present in a `rails.config` object, in
 * canonical order, with any unknown keys appended so nothing is dropped.
 */
export const listConfiguredDetectors = (rails: Rails | undefined): string[] => {
  const config = rails?.config;
  if (!config) return [];
  const present = Object.entries(config)
    .filter(([, value]) => value != null)
    .map(([key]) => key);
  const known = DETECTOR_ORDER.filter((key) => present.includes(key));
  const unknown = present.filter((key) => !DETECTOR_ORDER.includes(key));
  return [...known, ...unknown];
};

const stageFlows = (rails: Rails | undefined, scope: Scope): string[] => {
  switch (scope) {
    case 'input':
      return rails?.input?.flows ?? [];
    case 'output':
      return rails?.output?.flows ?? [];
    case 'retrieval':
      return rails?.retrieval?.flows ?? [];
    case 'tool_input':
      return rails?.tool_input?.flows ?? [];
    case 'tool_output':
      return rails?.tool_output?.flows ?? [];
  }
};

const SCOPE_ORDER: Scope[] = ['input', 'output', 'retrieval', 'tool_input', 'tool_output'];

/**
 * Derive the stages a detector runs at, from two signals unioned together:
 *  1. the detector's own `input`/`output`/`retrieval` sub-config (structural), and
 *  2. flows that reference it, matched via the flow recognition registry.
 */
export const deriveScopes = (rails: Rails | undefined, key: string): Scope[] => {
  const scopes = new Set<Scope>();

  const detector = rails?.config?.[key as DetectorKey];
  if (isPlainObject(detector)) {
    for (const scope of ['input', 'output', 'retrieval'] as const) {
      if (detector[scope] != null) scopes.add(scope);
    }
  }

  for (const scope of SCOPE_ORDER) {
    for (const flow of stageFlows(rails, scope)) {
      if (recognizeFlow(flow).detectorKey === key) {
        scopes.add(scope);
        break;
      }
    }
  }

  return SCOPE_ORDER.filter((scope) => scopes.has(scope));
};

// Match secret-bearing keys as whole words or snake_case segments, so prefixed
// keys like `nim_api_key`, `hf_token`, and `access_token` are masked too. The
// trailing `(?:_|$)` keeps non-secrets like `max_tokens` unmasked.
const SECRET_KEY =
  /(?:^|_)(?:api_?key|secret|token|password|passwd|credential|private_?key)(?:_|$)/i;

const formatScalar = (key: string, value: string | number | boolean): string => {
  if (SECRET_KEY.test(key)) return '••••••••';
  return String(value);
};

/**
 * Produce a faithful, shallow label/value summary of a detector's config. Scalar
 * fields render directly; scoped option objects (`{ entities }`, `{ patterns }`)
 * and simple `{ enabled }` toggles are summarized; secrets are masked. The full
 * object is always available via the raw-config section.
 */
export const summarizeDetector = (value: unknown): Field[] => {
  if (!isPlainObject(value)) return [];
  const fields: Field[] = [];

  for (const [key, raw] of Object.entries(value)) {
    if (raw == null) continue;
    const label = humanize(key);

    if (Array.isArray(raw)) {
      const items = raw.map((item) => (isPlainObject(item) ? JSON.stringify(item) : String(item)));
      fields.push({ label, value: items.length ? items.join(', ') : '—' });
      continue;
    }

    if (isPlainObject(raw)) {
      if (Array.isArray(raw.entities)) {
        fields.push({
          label: `${label} entities`,
          value: (raw.entities as string[]).join(', ') || '—',
        });
      } else if (Array.isArray(raw.patterns)) {
        fields.push({
          label: `${label} patterns`,
          value: `${(raw.patterns as string[]).length} pattern(s)`,
        });
      } else if (typeof raw.enabled === 'boolean') {
        fields.push({ label, value: raw.enabled ? 'Enabled' : 'Disabled' });
      } else {
        const keys = Object.keys(raw);
        fields.push({ label, value: keys.length ? keys.map(humanize).join(', ') : '—' });
      }
      continue;
    }

    fields.push({ label, value: formatScalar(key, raw as string | number | boolean) });
  }

  return fields;
};
