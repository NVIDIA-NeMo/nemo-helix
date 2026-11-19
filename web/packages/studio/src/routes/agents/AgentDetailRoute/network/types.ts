// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** A network rule in the OpenShell policy YAML shape. */
export interface NetworkRule {
  name?: string;
  endpoints?: { host?: string; port?: number; [key: string]: unknown }[];
  binaries?: { path?: string }[];
  [key: string]: unknown;
}

export type NetworkRequestStatus = 'pending' | 'approved' | 'rejected';

/** One connection a sandbox asked for. */
export interface NetworkRequest {
  id: string;
  /** The gateway may add statuses; anything unknown is shown as it is. */
  status: NetworkRequestStatus | string;
  rule_name: string;
  endpoints: string[];
  binary: string;
  rationale: string;
  security_notes: string;
  rejection_reason: string;
  hit_count: number;
  first_seen: string | null;
  last_seen: string | null;
  decided: string | null;
  rule: NetworkRule;
  /** Whether the agent already carries this rule. */
  remembered: boolean;
}

export interface AgentSandbox {
  sandbox: string;
  kind: 'deployment' | 'job' | 'sandbox';
  /** The deployment or job the sandbox belongs to. */
  name: string;
  phase: string;
  requests: NetworkRequest[];
  error: string;
}

export interface AgentNetworkPolicy {
  /** Rules the agent carries into every new sandbox, by rule name. */
  rules: Record<string, NetworkRule>;
  agent_yaml: string;
  sandboxes: AgentSandbox[];
  gateway_error: string;
}
