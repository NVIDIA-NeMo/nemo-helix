// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The host contract mirrors web/packages/studio/src/plugins/types.ts. This plugin
// calls only its own service, so the SDK handle stays opaque.
export interface PluginHost {
  workspaceId: string;
  /** Origin the platform API is served from; empty when same-origin. */
  apiBaseUrl: string;
  auth: { accessToken: string; getAccessToken: () => string };
  sdk: unknown;
  navigation: { navigate: (to: string) => void; back: () => void };
  notifications: {
    notify: (
      message: string,
      type?: 'success' | 'error' | 'info' | 'warning',
      options?: { durationMs?: number | false }
    ) => void;
  };
  telemetry: {
    info: (message: string, cause?: unknown) => void;
    warn: (message: string, cause?: unknown) => void;
    error: (message: string, cause?: unknown) => void;
    event: (name: string, attributes?: Record<string, unknown>) => void;
  };
  breadcrumbs: { set: (trail: { label: string; href?: string }[]) => void };
}

export interface PluginRootProps {
  host: PluginHost;
}

export interface PluginNavGroup {
  group: string;
  items: { id: string; iconName: string; label: string; href: string }[];
}

export type JobCollection = 'simulate' | 'export';

export interface Target {
  kind: 'agent' | 'model';
  agent?: string;
  model?: string;
  /** Keep the agent's own tools (its MCP servers) during the run. Agents only. */
  use_agent_tools?: boolean;
}

export interface SimulateRequest {
  target: Target;
  probes: string[];
  users: number;
  locale: string;
  max_turns?: number;
  /** Fileset of custom scenarios or topics, as `nemo user-sim run --scenarios` uploads. */
  scenarios?: string | null;
  seed?: number;
  preview?: string;
  exclude_personas?: string[];
  /** Run these conversations of an earlier run again, with the same personas and topics. */
  rerun?: { job: string; conversations: string[] };
}

/** A custom set uploaded with `nemo user-sim run --scenarios`. */
export interface ScenarioSet {
  fileset: string;
  kind: 'topics' | 'tool requests';
  probes: string[];
  /** How many topics a topic set holds. */
  size: number | null;
}

export interface ExportRequest {
  jobs: string[];
  fileset: string;
  quality: 'strict' | 'generous';
  require_review: boolean;
}

export interface Job<Spec = SimulateRequest> {
  name: string;
  created_at?: string;
  status?: string;
  spec: Spec;
  error_details?: { message?: string } | null;
}

/** One entry of a simulate job's `summary` result. */
export interface Conversation {
  trajectory_id: string;
  status: 'ok' | 'failed' | 'running';
  error: { category: string; message: string } | null;
  turns: number | null;
  persona_name?: string;
  persona_age_bin?: string;
  persona_education_level?: string;
  persona_occupation?: string;
  user_interaction_style?: string;
  probe_type?: string;
  /** IDs of the calls the agent made to its own tools, whose results are real. */
  own_tool_calls?: string[];
}

export interface Summary {
  /** What the run tested and the model behind it. Absent from runs that started before they were recorded. */
  target?: { kind: 'agent' | 'model'; agent?: string; model?: string; use_agent_tools?: boolean };
  assistant_model?: string;
  totals: { conversations: number; completed: number; failed: number; running?: number };
  conversations: Conversation[];
}

/** A simulate job's `scores` result, keyed by trajectory ID. */
export type Scores = Record<
  string,
  {
    passed: boolean;
    quality: number | null;
    /** The failed axes, comma separated. */
    reason: string | null;
    /** The tool-use scenario the conversation ran, when its probe has an answer key. */
    tested?: Tested | null;
    /** A plain sentence per failed axis. */
    why?: Record<string, string>;
    /** Which of the scenario's two tools it called, named by the scenario set's own labels. */
    picked?: Picked | null;
  }
>;

export interface Picked {
  pick: 'should_call' | 'should_not_call';
  label: string;
  tool: string;
}

export interface Tested {
  expected: string | null;
  should_call: string | null;
  should_not_call: string | null;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: string;
  tool_calls?: { id?: string; function: { name: string; arguments: string } }[];
  tool_call_id?: string;
}

/** A simulate job's `transcripts` result: each finished conversation's messages, keyed by trajectory ID. */
export type Transcripts = Record<string, ChatMessage[]>;

/** One entry of the plugin's probe catalog (``GET /apis/user-sim/v1/probes``). */
export interface Probe {
  name: string;
  label: string;
  tests: string;
  suits: string;
}

export interface ProbeSuggestions {
  agent: string;
  probes: { name: string; reason: string }[];
  /** The agent files the suggestion was inferred from. */
  read: string[];
}

/** Who one simulated user is, from a persona preview. */
export interface PersonaCard {
  persona_uuid: string;
  name: string;
  age: string;
  education: string;
  occupation: string;
  location: string | null;
  style: string;
  probe: string;
  traits: Record<string, number | null>;
}

export interface PersonaPreview {
  preview: string;
  personas: PersonaCard[];
}
