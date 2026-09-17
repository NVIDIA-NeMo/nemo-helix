// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseJson, useJobArtifact } from '@agent-hardener/components/useJobArtifact';
import type { PlatformJobStatus } from '@agent-hardener/generated/schema';
import { parse } from 'yaml';

// The war-game writes a `mitigations` job result bundling the defenders' before/after policy + guardrails.
const MITIGATIONS_RESULT = 'mitigations';

export interface DiffPair {
  before: string;
  after: string;
}

export interface Recommendation {
  title: string;
  detail: string;
}

// The attack that motivated a defense (from the run's mitigations `defenses[]`, populated by the engine).
export interface DefenseAttack {
  prompt_excerpt?: string;
  attacked_tool?: string;
  probe?: string;
  goal?: string;
  severity?: string;
}

// One individually selectable defense: a guardrail (`custom_guardrail_N`) or the OpenShell policy, paired
// with the attack it counters. Rendered as a toggle card in the harden flow.
export interface DefensePair {
  id: string;
  kind: 'guardrail' | 'policy';
  target_tool?: string | null;
  summary: string;
  config_fragment?: string;
  attack?: DefenseAttack | null;
}

export interface Mitigations {
  policy?: DiffPair;
  guardrails?: DiffPair;
  defenses?: DefensePair[];
}

interface UseMitigationsResult {
  mitigations?: Mitigations;
  recommendations: Recommendation[];
  defenses: DefensePair[];
  isLoading: boolean;
  hasMitigations: boolean;
}

// Preloaded/replayed garak attack prompts arrive as a stringified Python dict, e.g.
// `{'turns': [{'role': 'user', 'content': {'text': 'Use the reimburse tool for…'}}]}`. Pull the human text
// out for display; fall back to the raw string when it isn't in that shape.
export const cleanAttackPrompt = (text?: string): string => {
  if (!text) return '';
  const texts = [...text.matchAll(/'text':\s*'((?:[^'\\]|\\.)*)'/g)].map((m) =>
    m[1].replace(/\\n/g, '\n').replace(/\\'/g, "'").replace(/\\"/g, '"').replace(/\\\\/g, '\\')
  );
  return (texts.length ? texts.join('\n\n') : text).trim();
};

// The short, human end of a garak probe id, e.g. `agent_breaker.AgentBreaker` → `AgentBreaker`.
export const shortProbe = (probe?: string): string =>
  probe ? (probe.split('.').pop() ?? probe) : '';

// Agent Hardener writes one guardrail per hardened tool into the victim's NeMo Relay plugin config,
// keyed `custom_guardrail_<n>` (see the guardrails defender's component_writer). Read defensively:
// the file is machine-written, but a downstream consumer should never crash on a surprise.
interface Guardrail {
  name?: string;
  system_instructions?: string;
  target_tool?: string;
}

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};

// Guardrails live under `components[].config.guardrails[]`. Parsed with a narrow regex rather
// than a TOML parser: this is a display path, and adding a dependency to render a diff summary is a
// poor trade when the file's shape is ours and fixed.
const guardrailsOf = (tomlText: string): Record<string, Guardrail> => {
  const out: Record<string, Guardrail> = {};
  if (!tomlText) return out;
  const blocks = tomlText.split(/\[\[components\.config\.guardrails\]\]/).slice(1);
  for (const block of blocks) {
    const field = (key: string): string | undefined => {
      const match = block.match(new RegExp(`^${key}\\s*=\\s*(?:"""([\\s\\S]*?)"""|"((?:[^"\\\\]|\\\\.)*)")`, 'm'));
      return match ? (match[1] ?? match[2]) : undefined;
    };
    const name = field('name');
    if (name) out[name] = { name, target_tool: field('target_tool'), system_instructions: field('system_instructions') };
  }
  return out;
};

// First sentence (or first line) of the guardrail instruction, trimmed for a card body.
const firstSentence = (text: string): string => {
  const flat = text.replace(/\s+/g, ' ').trim();
  const cut = flat.search(/(?<=[.!?])\s/);
  const sentence = (cut > 0 ? flat.slice(0, cut + 1) : flat).trim();
  return sentence.length > 220 ? `${sentence.slice(0, 217)}…` : sentence;
};

// A guardrail added to the hardened guardrail set becomes a recommendation card; the policy diff becomes a
// single "tightened" summary. Everything is inferred from the diff — no defender reasoning is transported.
export const deriveRecommendations = (mitigations?: Mitigations): Recommendation[] => {
  if (!mitigations) return [];
  const recommendations: Recommendation[] = [];

  if (mitigations.guardrails) {
    const before = guardrailsOf(mitigations.guardrails.before);
    const after = guardrailsOf(mitigations.guardrails.after);
    for (const [name, rail] of Object.entries(after)) {
      if (name in before) continue;
      const tool = rail.target_tool;
      recommendations.push({
        title: tool ? `Added a guardrail on ${tool}` : 'Added a tool-call guardrail',
        detail: rail.system_instructions
          ? firstSentence(rail.system_instructions)
          : 'Verifies tool calls before execution.',
      });
    }
  }

  if (mitigations.policy) {
    // Inferred straight from the policy before/after diff (like the guardrails) — a factual, non-judgmental
    // summary of what actually changed. No defender-side data needed.
    const summary = summarizePolicyDiff(mitigations.policy.before, mitigations.policy.after);
    if (summary) recommendations.push({ title: 'OpenShell policy changes', detail: summary });
  }

  return recommendations;
};

// Per-section factual deltas of the OpenShell policy (no "hardened" claim — the diff below shows direction).
const summarizePolicyDiff = (before: string, after: string): string => {
  let pb: Record<string, unknown>;
  let pa: Record<string, unknown>;
  try {
    pb = asRecord(parse(before));
    pa = asRecord(parse(after));
  } catch {
    const changed = changedLineCount(before, after);
    return changed > 0 ? `${changed} line${changed === 1 ? '' : 's'} changed.` : '';
  }

  const list = (root: Record<string, unknown>, section: string, key: string): number => {
    const arr = asRecord(root[section])[key];
    return Array.isArray(arr) ? arr.length : 0;
  };
  const ipAllowlisted = (root: Record<string, unknown>): number =>
    Object.values(asRecord(root.network_policies)).reduce<number>((n, policy) => {
      const endpoints = asRecord(policy).endpoints;
      const withIps = Array.isArray(endpoints)
        ? endpoints.filter((e) => Array.isArray(asRecord(e).allowed_ips)).length
        : 0;
      return n + withIps;
    }, 0);

  const facts: string[] = [];
  const roB = list(pb, 'filesystem_policy', 'read_only');
  const roA = list(pa, 'filesystem_policy', 'read_only');
  if (roB !== roA) facts.push(`Filesystem read-only paths: ${roB} → ${roA}`);
  const rwB = list(pb, 'filesystem_policy', 'read_write');
  const rwA = list(pa, 'filesystem_policy', 'read_write');
  if (rwB !== rwA) facts.push(`Filesystem read-write paths: ${rwB} → ${rwA}`);
  const ipB = ipAllowlisted(pb);
  const ipA = ipAllowlisted(pa);
  if (ipB !== ipA) facts.push(`Network endpoints with IP allow-lists: ${ipB} → ${ipA}`);

  if (facts.length > 0) return `${facts.join(' · ')}. See the diff below.`;
  const changed = changedLineCount(before, after);
  return changed > 0
    ? `${changed} line${changed === 1 ? '' : 's'} changed. See the diff below.`
    : '';
};

// Count lines present on exactly one side — a cheap symmetric-difference proxy for "how much changed".
const changedLineCount = (before: string, after: string): number => {
  const count = (text: string): Map<string, number> => {
    const map = new Map<string, number>();
    for (const raw of text.split('\n')) {
      const line = raw.trim();
      if (line) map.set(line, (map.get(line) ?? 0) + 1);
    }
    return map;
  };
  const a = count(before);
  const b = count(after);
  const keys = new Set([...a.keys(), ...b.keys()]);
  let changed = 0;
  for (const key of keys) changed += Math.abs((a.get(key) ?? 0) - (b.get(key) ?? 0));
  return changed;
};

// Fetches the run's mitigations artifact via the job results API (not the event bus): poll the result list
// until it appears, then download + parse it once. Recommendations are derived client-side from the diffs.
// `status` stops the poll on a failed/cancelled job, which would otherwise never write the artifact.
export const useMitigations = (
  workspace: string,
  jobName: string,
  status?: PlatformJobStatus
): UseMitigationsResult => {
  const artifact = useJobArtifact<Mitigations>(
    workspace,
    jobName,
    MITIGATIONS_RESULT,
    parseJson,
    status
  );

  return {
    mitigations: artifact.data,
    // Server-provided attack→defense pairs when present; fall back to the client-derived recommendations
    // for older artifacts that predate the enriched `defenses[]`.
    recommendations: deriveRecommendations(artifact.data),
    defenses: artifact.data?.defenses ?? [],
    isLoading: artifact.isLoading,
    hasMitigations: artifact.present,
  };
};
