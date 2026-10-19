// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';

// Registry of canned example agents. Each entry references curated static assets
// under public/sample-agents/<dir>/ by path (fetched on demand, never bundled) —
// mirroring src/constants/sampleDatasets.ts. Used by the Create Example Agent
// modal (fetch + parse agent.yml, inject model, POST).
//
// Agent configs only. Eval configs and datasets are NOT seeded from here: Studio's
// Run Evaluation flow takes a user-supplied config and dataset, so the sample eval
// artifacts live with the example they score, at
// plugins/nemo-agents/examples/nemo-agent-config/email-security-triage/.
//
// INVARIANT: a sample can depend on something being installed in the deploy
// venv/image, or it fails at startup:
//
// - Fabric (`nemo-agents-spec-v1`) entries need each `mcp.servers.<n>.url`
//   console script on PATH, since Fabric spawns it as a stdio MCP child. No
//   shipped sample declares `mcp:` today, so nothing currently relies on this.
//
// Each public/sample-agents/<dir>/agent.yml is an independent copy of the
// example's config; keep them in sync by hand.
export interface SampleAgent {
  key: string;
  displayName: string;
  /** Prefix for generated agent names; drives onboarding detection. */
  namePrefix: string;
  /** Public path to the agent config (parsed + model-injected at create). */
  agentConfigPath: string;
  /** Config format identifier sent to the create API. Defaults to
   *  `nat-workflow-v1` server-side when omitted; set to `nemo-agents-spec-v1`
   *  for Fabric-backed samples so the API validates them as Fabric, not NAT. */
  configFormat?: string;
}

export const SAMPLE_AGENTS: SampleAgent[] = [
  {
    key: 'email_security_triage',
    displayName: 'Email Security Triage',
    namePrefix: 'email-security-triage',
    agentConfigPath: 'sample-agents/email-security-triage/agent.yml',
    configFormat: 'nemo-agents-spec-v1',
  },
];

export const DEFAULT_SAMPLE_AGENT_KEY = SAMPLE_AGENTS[0].key;

export const getSampleAgent = (key: string): SampleAgent =>
  SAMPLE_AGENTS.find((agent) => agent.key === key) ?? SAMPLE_AGENTS[0];

export const buildSampleAgentName = (namePrefix: string): string =>
  `${namePrefix}-${Math.random().toString(36).slice(2, 8)}`;

export const isSampleAgentName = (name: string): boolean =>
  SAMPLE_AGENTS.some((agent) => name.startsWith(`${agent.namePrefix}-`));

/**
 * Infer which sample-agent example a deployed agent came from by matching its
 * generated name (`${namePrefix}-<suffix>`). Returns the example key, or
 * undefined for agents not created from an example.
 *
 * Robustness: requires the `${namePrefix}-` separator (so a prefix only matches
 * a real name boundary, not a partial token) and picks the LONGEST matching
 * prefix — so when one prefix is a substring of another (e.g. "test-" vs
 * "test-agent-"), the most specific wins deterministically.
 */
export const sampleAgentKeyForAgentName = (name: string | undefined): string | undefined => {
  if (!name) return undefined;
  return SAMPLE_AGENTS.filter((agent) => name.startsWith(`${agent.namePrefix}-`)).sort(
    (a, b) => b.namePrefix.length - a.namePrefix.length
  )[0]?.key;
};

export const sampleAgentFormSchema = z.object({
  exampleKey: z.string().min(1, 'Example is required'),
  modelName: z.string().min(1, 'Model is required'),
});
