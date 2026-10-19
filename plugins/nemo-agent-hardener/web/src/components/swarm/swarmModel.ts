// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The fixed Agent Hardener topology + a reducer that turns the live EventBus stream into per-node status
// and run progress. Node positions are authored (a faithful port of the demo's hand-laid SVG), not
// force-simulated. Roles mirror agent-hardener's attacker/defender/victim/validator swarms.

import type { SwarmEvent } from '@agent-hardener/components/eventTypes';

export type NodeGroup =
  | 'analyzer'
  | 'attacker'
  | 'defender'
  | 'victim'
  | 'validator'
  | 'update'
  | 'summary';
export type NodeStatus = 'pending' | 'running' | 'success' | 'blocked' | 'failed';

export interface SwarmNode {
  id: string;
  title: string;
  group: NodeGroup;
  x: number;
  y: number;
  isManager?: boolean;
}

export interface SwarmEdge {
  from: string;
  to: string;
}

export const GROUP_COLOR: Record<NodeGroup, string> = {
  analyzer: '#c855fa',
  attacker: '#ff3855',
  defender: '#00e676',
  victim: '#448aff',
  validator: '#ffab40',
  update: '#29b6f6',
  summary: '#cfd8dc',
};

// Authored layout on a 1000x720 canvas: attacker swarm (left), sandbox (center), defender swarm
// (right), validator swarm (bottom).
export const NODES: SwarmNode[] = [
  { id: 'benign_analyzer', title: 'Benign Analyzer', group: 'analyzer', x: 170, y: 430 },
  {
    id: 'attacker_manager',
    title: 'Attacker Manager',
    group: 'attacker',
    x: 190,
    y: 150,
    isManager: true,
  },
  { id: 'attacker', title: 'Attacker', group: 'attacker', x: 190, y: 290 },
  { id: 'victim_agent', title: 'Victim Agent', group: 'victim', x: 500, y: 300 },
  {
    id: 'defender_manager',
    title: 'Defender Manager',
    group: 'defender',
    x: 810,
    y: 150,
    isManager: true,
  },
  { id: 'guardrails_defender', title: 'Guardrails Defender', group: 'defender', x: 730, y: 290 },
  { id: 'openshell_defender', title: 'OpenShell Defender', group: 'defender', x: 890, y: 290 },
  { id: 'update_victim_agent_policy', title: 'Deploy Agent', group: 'update', x: 810, y: 430 },
  {
    id: 'validator_manager',
    title: 'Validator Manager',
    group: 'validator',
    x: 440,
    y: 600,
    isManager: true,
  },
  { id: 'attacker_validator', title: 'Attacker Validator', group: 'validator', x: 330, y: 660 },
  { id: 'benign_validator', title: 'Benign Validator', group: 'validator', x: 550, y: 660 },
  { id: 'summary', title: 'Summary', group: 'summary', x: 810, y: 600 },
];

export const EDGES: SwarmEdge[] = [
  { from: 'attacker_manager', to: 'attacker' },
  { from: 'attacker', to: 'victim_agent' },
  { from: 'benign_analyzer', to: 'victim_agent' },
  { from: 'defender_manager', to: 'guardrails_defender' },
  { from: 'defender_manager', to: 'openshell_defender' },
  { from: 'defender_manager', to: 'update_victim_agent_policy' },
  { from: 'guardrails_defender', to: 'victim_agent' },
  { from: 'openshell_defender', to: 'victim_agent' },
  { from: 'update_victim_agent_policy', to: 'victim_agent' },
  { from: 'victim_agent', to: 'validator_manager' },
  { from: 'validator_manager', to: 'attacker_validator' },
  { from: 'validator_manager', to: 'benign_validator' },
  { from: 'validator_manager', to: 'summary' },
];

// Nodes activated when a stage phase starts/completes (payload.phase from agent-hardener's stages).
const PHASE_NODES: Record<string, string[]> = {
  attackers: ['attacker_manager', 'attacker'],
  defenders: ['defender_manager', 'guardrails_defender', 'openshell_defender'],
  victim: ['victim_agent'],
  validators: ['validator_manager', 'attacker_validator', 'benign_validator'],
};

// Map an agent event to a graph node. agent-hardener's `agent_name`/`agent_id` (e.g. `garak-agent-breaker`,
// `defender-guardrails`, `openshell-policy-defender`) don't equal node ids, so resolve by role — and, for
// leaf agents, by name substring (defenders) or `validator_kind` (validators). Managers/victim still match
// by exact title. Getting this right is what makes clicking a leaf agent show its own activity + prompts.
const normalizeKey = (value: string): string =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
const NODE_BY_TITLE: Record<string, string> = Object.fromEntries(
  NODES.map((n) => [normalizeKey(n.title), n.id])
);
const ROLE_MANAGER: Record<string, string> = {
  attacker: 'attacker_manager',
  defender: 'defender_manager',
  validator: 'validator_manager',
};

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined;

export const nodeForAgent = (payload: Record<string, unknown>): string | undefined => {
  const name = asString(payload.agent_name);
  if (name && NODE_BY_TITLE[normalizeKey(name)]) return NODE_BY_TITLE[normalizeKey(name)];
  const key = name ? normalizeKey(name) : '';
  const role = asString(payload.agent_role);
  const kind = asString(payload.validator_kind);
  switch (role) {
    case 'attacker':
      return 'attacker';
    case 'victim':
      return 'victim_agent';
    case 'defender':
      if (key.includes('guardrail')) return 'guardrails_defender';
      if (key.includes('openshell') || key.includes('policy')) return 'openshell_defender';
      return 'defender_manager';
    case 'validator':
      if (kind === 'attack') return 'attacker_validator';
      if (kind === 'benign') return 'benign_validator';
      return 'validator_manager';
    default:
      return role ? ROLE_MANAGER[role] : undefined;
  }
};

// One agent-activity line (from the agent lifecycle events), shown in the node's Activity log.
export interface LogEntry {
  ts?: number;
  label: string;
  text: string;
  level: 'info' | 'error';
}

// One prompt<->response transcript row (from `agent_exchange`), shown in the node's Prompts transcript.
// `blocked` is set for validator checks so the UI can mark a prompt allowed vs blocked.
export interface Exchange {
  ts?: number;
  request: string;
  response: string;
  label?: string;
  ok: boolean;
  blocked?: boolean;
}

// One internal LLM request<->completion (from `llm_call`), shown in the node's LLM-calls list.
export interface LlmCall {
  ts?: number;
  request: string;
  response: string;
  label?: string;
  ok: boolean;
}

export interface SwarmState {
  statuses: Record<string, NodeStatus>;
  phase: string;
  round: number;
  finalPass: boolean;
  nodeLogs: Record<string, LogEntry[]>;
  nodeExchanges: Record<string, Exchange[]>;
  nodeLlmCalls: Record<string, LlmCall[]>;
}

// Fold the ordered event stream into node statuses + per-node logs/exchanges + run progress. Deterministic:
// replaying the same prefix always yields the same state, so late SSE subscribers converge correctly.
export const deriveSwarmState = (events: SwarmEvent[]): SwarmState => {
  const statuses: Record<string, NodeStatus> = Object.fromEntries(
    NODES.map((n) => [n.id, 'pending'])
  );
  const nodeLogs: Record<string, LogEntry[]> = {};
  const nodeExchanges: Record<string, Exchange[]> = {};
  const nodeLlmCalls: Record<string, LlmCall[]> = {};
  let phase = '';
  let round = 0;
  let finalPass = false;

  const setGroup = (ids: string[] | undefined, status: NodeStatus) =>
    ids?.forEach((id) => (statuses[id] = status));
  const pushLog = (id: string, entry: LogEntry) => (nodeLogs[id] ??= []).push(entry);
  const pushExchange = (id: string, ex: Exchange) => (nodeExchanges[id] ??= []).push(ex);
  const pushLlmCall = (id: string, call: LlmCall) => (nodeLlmCalls[id] ??= []).push(call);

  for (const ev of events) {
    const { event, payload } = ev;
    const phaseName = asString(payload.phase);
    switch (event) {
      case 'phase_started':
        setGroup(phaseName ? PHASE_NODES[phaseName] : undefined, 'running');
        if (phaseName) phase = phaseName;
        break;
      case 'phase_completed':
        setGroup(phaseName ? PHASE_NODES[phaseName] : undefined, 'success');
        break;
      case 'agent_started': {
        const id = nodeForAgent(payload);
        if (id) {
          statuses[id] = 'running';
          pushLog(id, {
            ts: ev.ts,
            label: 'started',
            text: asString(payload.agent_name) ?? '',
            level: 'info',
          });
        }
        break;
      }
      case 'agent_progress': {
        const id = nodeForAgent(payload);
        const message = asString(payload.message);
        if (id && message)
          pushLog(id, { ts: ev.ts, label: 'progress', text: message, level: 'info' });
        break;
      }
      case 'agent_completed': {
        const id = nodeForAgent(payload);
        if (id) {
          statuses[id] = payload.ok === false ? 'failed' : 'success';
          const seconds =
            typeof payload.duration_seconds === 'number'
              ? ` (${payload.duration_seconds.toFixed(1)}s)`
              : '';
          pushLog(id, {
            ts: ev.ts,
            label: `completed${seconds}`,
            text: asString(payload.summary) ?? '',
            level: 'info',
          });
        }
        break;
      }
      case 'agent_failed': {
        const id = nodeForAgent(payload);
        if (id) {
          statuses[id] = 'failed';
          pushLog(id, {
            ts: ev.ts,
            label: 'failed',
            text: asString(payload.error) ?? '',
            level: 'error',
          });
        }
        break;
      }
      case 'agent_exchange': {
        const id = nodeForAgent(payload);
        const exchange: Exchange = {
          ts: ev.ts,
          request: asString(payload.request) ?? '',
          response: asString(payload.response) ?? '',
          label: asString(payload.label),
          ok: payload.ok !== false,
          blocked: typeof payload.blocked === 'boolean' ? payload.blocked : undefined,
        };
        if (id) pushExchange(id, exchange);
        // The victim is the target of attacker/benign prompts — aggregate those onto it too.
        const role = asString(payload.agent_role);
        if (
          id !== 'victim_agent' &&
          (role === 'attacker' || role === 'benign' || role === 'validator')
        ) {
          pushExchange('victim_agent', exchange);
        }
        break;
      }
      case 'llm_call': {
        const id = nodeForAgent(payload);
        if (id) {
          pushLlmCall(id, {
            ts: ev.ts,
            request: asString(payload.request) ?? '',
            response: asString(payload.response) ?? '',
            label: asString(payload.label),
            ok: payload.ok !== false,
          });
        }
        break;
      }
      case 'synth_phase':
      case 'interview_started':
        statuses.benign_analyzer = 'running';
        break;
      case 'interview_completed':
        statuses.benign_analyzer = 'success';
        break;
      case 'victim_control_started':
      case 'openshell_upload':
      case 'relay_policy_upload':
        statuses.update_victim_agent_policy = 'running';
        statuses.victim_agent = 'running';
        break;
      case 'victim_control_completed':
        statuses.update_victim_agent_policy = 'success';
        statuses.victim_agent = 'success';
        break;
      case 'round_started':
        round = typeof payload.round === 'number' ? payload.round : round + 1;
        break;
      case 'report_written':
        statuses.summary = 'success';
        break;
      case 'round_completed':
        if (payload.success === true) {
          finalPass = true;
          phase = 'FINAL PASS';
          statuses.summary = 'success';
        }
        break;
      default:
        break;
    }
  }

  return { statuses, phase, round, finalPass, nodeLogs, nodeExchanges, nodeLlmCalls };
};

// One benign-suite recon step (from `synth_phase`), shown as a checklist in the manifest generate flow.
export interface ReconStep {
  phase: string;
  label: string;
}

// Fold `synth_phase` events into an ordered recon checklist. agent-hardener emits one event per completed recon
// node, so every received step is done; the consumer shows a trailing "working" affordance while the job runs.
// Deduped by phase, preserving first-seen order.
export const reconSteps = (events: SwarmEvent[]): ReconStep[] => {
  const byPhase = new Map<string, ReconStep>();
  for (const ev of events) {
    if (ev.event !== 'synth_phase') continue;
    const phase = asString(ev.payload.phase);
    if (!phase) continue;
    byPhase.set(phase, { phase, label: asString(ev.payload.label) ?? phase });
  }
  return [...byPhase.values()];
};

// The current lifecycle stage the run is in, from the latest status/deploy event — a live "what's happening
// now" label (e.g. "Building and starting sandbox", "Waiting for victim health", "benign recon") so the
// generate flow can show real progress instead of a generic spinner. Recon detail is listed separately.
export const currentActivity = (events: SwarmEvent[]): string | undefined => {
  let label: string | undefined;
  for (const ev of events) {
    if (ev.event === 'status_started') {
      label = asString(ev.payload.label) ?? label;
    } else if (ev.event === 'victim_control_started') {
      label = 'Deploying victim…';
    } else if (ev.event === 'victim_control_completed') {
      label = 'Victim deployed';
    }
  }
  return label;
};
