// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Mirror of agent_hardener/events.py (the EventBus source of truth). Keep in sync with that catalog:
// the war-game POSTs these event names to the plugin's SSE relay, and this module decides how the
// Hardening tab renders each one.

export const EVENT_TYPES = [
  'status_started',
  'status_completed',
  'output',
  'round_started',
  'round_completed',
  'iteration_started',
  'iteration_completed',
  'report_written',
  'phase_started',
  'phase_completed',
  'victim_control_started',
  'victim_control_completed',
  'openshell_upload',
  'relay_policy_upload',
  'victim_warning',
  'preloaded_attacks_loaded',
  'attackers_completed',
  'attack_summary',
  'artifact_written',
  'attacker_summaries_prepared',
  'defender_summary',
  'policy_patches_aggregated',
  'agent_started',
  'agent_progress',
  'agent_completed',
  'agent_failed',
  'agent_exchange',
  'llm_call',
  'synth_phase',
  'interview_started',
  'interview_completed',
] as const;

export type EventType = (typeof EVENT_TYPES)[number];

export type EventCategory =
  | 'lifecycle'
  | 'round'
  | 'phase'
  | 'deploy'
  | 'attack'
  | 'defense'
  | 'agent'
  | 'synth';

const CATEGORY_BY_EVENT: Record<EventType, EventCategory> = {
  status_started: 'lifecycle',
  status_completed: 'lifecycle',
  output: 'lifecycle',
  round_started: 'round',
  round_completed: 'round',
  iteration_started: 'round',
  iteration_completed: 'round',
  report_written: 'round',
  phase_started: 'phase',
  phase_completed: 'phase',
  victim_control_started: 'deploy',
  victim_control_completed: 'deploy',
  openshell_upload: 'deploy',
  relay_policy_upload: 'deploy',
  victim_warning: 'deploy',
  preloaded_attacks_loaded: 'attack',
  attackers_completed: 'attack',
  attack_summary: 'attack',
  artifact_written: 'attack',
  attacker_summaries_prepared: 'defense',
  defender_summary: 'defense',
  policy_patches_aggregated: 'defense',
  agent_started: 'agent',
  agent_progress: 'agent',
  agent_completed: 'agent',
  agent_failed: 'agent',
  agent_exchange: 'agent',
  llm_call: 'agent',
  synth_phase: 'synth',
  interview_started: 'synth',
  interview_completed: 'synth',
};

export const categoryOf = (event: string): EventCategory | undefined =>
  (CATEGORY_BY_EVENT as Record<string, EventCategory>)[event];

// One event as relayed by the plugin SSE endpoint (`{event, payload}`). `ts` is the client receive time
// (the wire carries no timestamp), stamped once on arrival so it's stable across re-renders.
export interface SwarmEvent {
  id: number;
  event: string;
  payload: Record<string, unknown>;
  ts?: number;
}
