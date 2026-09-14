// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SwarmEvent } from '@agent-hardener/components/eventTypes';
import { deriveSwarmState, reconSteps } from '@agent-hardener/components/swarm/swarmModel';

const evt = (event: string, payload: Record<string, unknown> = {}, id = 0): SwarmEvent => ({
  id,
  event,
  payload,
});

describe('reconSteps', () => {
  it('collects synth_phase labels in order, deduped by phase, ignoring other events', () => {
    const steps = reconSteps([
      evt('synth_phase', { phase: 'nl_parser', label: 'parsed capabilities' }),
      evt('agent_started', { agent_name: 'x' }),
      evt('synth_phase', { phase: 'api_prober', label: 'probed the endpoint' }),
      evt('synth_phase', { phase: 'nl_parser', label: 'parsed capabilities' }),
    ]);
    expect(steps.map((s) => s.phase)).toEqual(['nl_parser', 'api_prober']);
    expect(steps[0]?.label).toBe('parsed capabilities');
  });

  it('falls back to the phase name when a label is missing and skips phase-less events', () => {
    const steps = reconSteps([evt('synth_phase', { phase: 'critic' }), evt('synth_phase', {})]);
    expect(steps).toEqual([{ phase: 'critic', label: 'critic' }]);
  });
});

describe('deriveSwarmState', () => {
  it('starts all nodes pending', () => {
    const { statuses, phase, round, finalPass } = deriveSwarmState([]);
    expect(statuses.victim_agent).toBe('pending');
    expect(phase).toBe('');
    expect(round).toBe(0);
    expect(finalPass).toBe(false);
  });

  it('activates a phase group on phase_started and resolves it on phase_completed', () => {
    const running = deriveSwarmState([evt('phase_started', { phase: 'attackers' })]);
    expect(running.statuses.attacker_manager).toBe('running');
    expect(running.phase).toBe('attackers');

    const done = deriveSwarmState([
      evt('phase_started', { phase: 'attackers' }),
      evt('phase_completed', { phase: 'attackers' }),
    ]);
    expect(done.statuses.attacker).toBe('success');
  });

  it('maps agents to leaf nodes by role, name, and validator_kind', () => {
    const { statuses, nodeLogs } = deriveSwarmState([
      // Real agent-hardener agent_name values that don't equal node titles.
      evt('agent_started', { agent_name: 'defender-guardrails', agent_role: 'defender' }),
      evt('agent_completed', {
        agent_name: 'openshell-policy-defender',
        agent_role: 'defender',
        ok: true,
      }),
      evt('agent_failed', {
        agent_name: 'garak-agent-breaker',
        agent_role: 'attacker',
        error: 'boom',
      }),
      evt('agent_started', {
        agent_name: 'attack-replay',
        agent_role: 'validator',
        validator_kind: 'attack',
      }),
    ]);
    expect(statuses.guardrails_defender).toBe('running');
    expect(statuses.openshell_defender).toBe('success');
    expect(statuses.attacker).toBe('failed');
    expect(statuses.attacker_validator).toBe('running');
    expect(nodeLogs.attacker?.[0]?.level).toBe('error');
  });

  it('falls back to the role manager node when a defender name is unrecognized', () => {
    const { statuses } = deriveSwarmState([
      evt('agent_started', { agent_name: 'some-custom-defender', agent_role: 'defender' }),
    ]);
    expect(statuses.defender_manager).toBe('running');
  });

  it('buckets agent_exchange onto the agent node and aggregates attacker prompts onto the victim', () => {
    const { nodeExchanges } = deriveSwarmState([
      evt('agent_exchange', {
        agent_name: 'garak-agent-breaker',
        agent_role: 'attacker',
        request: 'do X',
        response: 'refused',
        label: 'jailbreak',
        ok: true,
      }),
    ]);
    expect(nodeExchanges.attacker?.[0]?.request).toBe('do X');
    expect(nodeExchanges.attacker?.[0]?.response).toBe('refused');
    expect(nodeExchanges.victim_agent?.[0]?.request).toBe('do X');
  });

  it('records validator verdicts (blocked/allowed) and per-agent LLM calls', () => {
    const { nodeExchanges, nodeLlmCalls } = deriveSwarmState([
      evt('agent_exchange', {
        agent_role: 'validator',
        validator_kind: 'benign',
        request: 'what time is it?',
        response: 'I cannot help',
        label: 'refused',
        ok: false,
        blocked: true,
      }),
      evt('llm_call', {
        agent_role: 'defender',
        agent_name: 'defender-guardrails',
        request: 'analyze this',
        response: 'rule: ...',
        label: 'gpt',
        ok: true,
      }),
    ]);
    expect(nodeExchanges.benign_validator?.[0]?.blocked).toBe(true);
    expect(nodeLlmCalls.guardrails_defender?.[0]?.request).toBe('analyze this');
  });

  it('marks a final pass on a successful round', () => {
    const { finalPass, phase, round } = deriveSwarmState([
      evt('round_started', { round: 1 }),
      evt('round_completed', { success: true }),
    ]);
    expect(round).toBe(1);
    expect(finalPass).toBe(true);
    expect(phase).toBe('FINAL PASS');
  });
});
