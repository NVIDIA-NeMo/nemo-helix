// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  categoryOf,
  type EventCategory,
  type SwarmEvent,
} from '@agent-hardener/components/eventTypes';
import { NODES, nodeForAgent, type NodeGroup } from '@agent-hardener/components/swarm/swarmModel';
import { ACCENT } from '@agent-hardener/theme';
import { Flex, Text } from '@nvidia/foundations-react-core';
import { FC, useLayoutEffect, useMemo, useRef } from 'react';

interface MessageFeedProps {
  events: SwarmEvent[];
}

const CATEGORY_COLOR: Record<EventCategory, string> = {
  lifecycle: ACCENT.gray,
  round: ACCENT.blue,
  phase: ACCENT.teal,
  deploy: ACCENT.purple,
  attack: ACCENT.red,
  defense: ACCENT.green,
  agent: ACCENT.yellow,
  synth: ACCENT.teal,
};

// Per-swarm text color so an agent's feed lines match its node color in the graph.
const GROUP_COLOR: Record<NodeGroup, string> = {
  analyzer: ACCENT.purple,
  attacker: ACCENT.red,
  defender: ACCENT.green,
  victim: ACCENT.blue,
  validator: ACCENT.yellow,
  update: ACCENT.teal,
  summary: ACCENT.gray,
};
const GROUP_BY_NODE: Record<string, NodeGroup> = Object.fromEntries(
  NODES.map((n) => [n.id, n.group])
);

const str = (value: unknown): string | undefined => (typeof value === 'string' ? value : undefined);
const num = (value: unknown): number | undefined => (typeof value === 'number' ? value : undefined);
const humanize = (event: string): string =>
  event.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());

// Color an event's line by its agent's swarm (so it matches the graph), falling back to the event category.
const lineColor = (evt: SwarmEvent): string => {
  const id = nodeForAgent(evt.payload);
  if (id && GROUP_BY_NODE[id]) return GROUP_COLOR[GROUP_BY_NODE[id]];
  return CATEGORY_COLOR[categoryOf(evt.event) ?? 'lifecycle'];
};

// Fallback for unknown events: inline only scalar payload fields (never raw blobs), as `key=value`.
const scalars = (payload: Record<string, unknown>): string =>
  Object.entries(payload)
    .filter(([, v]) => typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean')
    .map(([k, v]) => `${k}=${String(v)}`)
    .join(' ');

// Turn one event into a human-readable feed line (no raw `line=` — `output` shows its console text directly).
const formatEvent = (evt: SwarmEvent): string => {
  const p = evt.payload;
  const agent = str(p.agent_name) ?? 'Agent';
  switch (evt.event) {
    case 'output':
      return str(p.line) ?? '';
    case 'status_started':
    case 'status_completed':
      return str(p.label) ?? humanize(evt.event);
    case 'phase_started':
      return `Phase started: ${str(p.phase) ?? ''}`;
    case 'phase_completed':
      return `Phase completed: ${str(p.phase) ?? ''}`;
    case 'agent_started':
      return `${agent} started`;
    case 'agent_progress':
      return `${agent}: ${str(p.message) ?? ''}`;
    case 'agent_completed': {
      const seconds = num(p.duration_seconds);
      return `${p.ok === false ? '✗' : '✓'} ${agent}${seconds !== undefined ? ` (${seconds.toFixed(1)}s)` : ''}`;
    }
    case 'agent_failed':
      return `✗ ${agent} failed: ${str(p.error) ?? ''}`;
    case 'agent_exchange':
      return `${agent} → victim${str(p.label) ? ` [${str(p.label)}]` : ''}${p.blocked === true ? ' (blocked)' : p.blocked === false ? ' (allowed)' : ''}`;
    case 'llm_call':
      return `${agent} · LLM call${str(p.label) ? ` (${str(p.label)})` : ''}`;
    case 'round_started':
      return `Round ${num(p.round) ?? ''} started`.trim();
    case 'round_completed':
      return p.success === true ? 'Round passed' : 'Round completed';
    case 'report_written':
      return 'Report written';
    case 'attack_summary':
      return `Attack summary (${Array.isArray(p.attacks) ? p.attacks.length : 0} attacker(s))`;
    case 'defender_summary':
      return `Defender summary (${Array.isArray(p.defenders) ? p.defenders.length : 0} defender(s))`;
    case 'synth_phase':
      return str(p.label) ?? 'Recon step';
    case 'interview_started':
      return 'Interview started';
    case 'interview_completed':
      return 'Interview completed';
    case 'victim_control_started':
      return 'Deploying victim…';
    case 'victim_control_completed':
      return 'Victim deployed';
    default: {
      const detail = scalars(p);
      return detail ? `${humanize(evt.event)} — ${detail}` : humanize(evt.event);
    }
  }
};

const formatTime = (ts?: number): string =>
  ts ? new Date(ts).toLocaleTimeString([], { hour12: false }) : '';

// Live tail of the run's EventBus (newest last), auto-scrolling to the bottom unless the user has scrolled
// up to read history. Highlights the agent(s) currently running and separates each line for readability.
export const MessageFeed: FC<MessageFeedProps> = ({ events }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);

  // Pin to the newest event before paint (no top-then-jump flash). Follows while the user is at the
  // bottom; pauses the moment they scroll up to read history (see onScroll), resumes when they return.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el && atBottomRef.current) el.scrollTop = el.scrollHeight;
  }, [events]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (el) atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  // Agents currently working: started and not yet completed/failed (ordered, deduped by node).
  const running = useMemo(() => {
    const active = new Map<string, string>();
    for (const evt of events) {
      const id = nodeForAgent(evt.payload);
      if (!id) continue;
      if (evt.event === 'agent_started') active.set(id, str(evt.payload.agent_name) ?? id);
      else if (evt.event === 'agent_completed' || evt.event === 'agent_failed') active.delete(id);
    }
    return Array.from(active.entries());
  }, [events]);

  if (events.length === 0) {
    return (
      <Text kind="body/regular/md" className="text-subtle">
        Waiting for live events…
      </Text>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {running.length > 0 ? (
        <Flex align="center" gap="density-xs" className="mb-2 shrink-0 flex-wrap">
          <Text kind="body/regular/sm" className="text-subtle">
            Now running:
          </Text>
          {running.map(([id, name]) => (
            <span
              key={id}
              className="rounded-full bg-surface-sunken px-2 py-0.5 text-xs font-medium"
              style={{ color: GROUP_COLOR[GROUP_BY_NODE[id]] ?? ACCENT.gray }}
            >
              {name}
            </span>
          ))}
        </Flex>
      ) : null}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="min-h-0 flex-1 overflow-auto pr-density-xs"
      >
        {events.map((evt) => {
          const nodeId = nodeForAgent(evt.payload);
          const isRunning = nodeId ? running.some(([id]) => id === nodeId) : false;
          return (
            <div
              key={evt.id}
              className={`flex items-baseline gap-2.5 border-t border-base px-1 py-1.5 ${isRunning ? 'bg-surface-sunken' : ''}`}
            >
              <span className="shrink-0 pt-1 text-xs tabular-nums text-subtle">
                {formatTime(evt.ts)}
              </span>
              <span
                className="shrink-0 text-xs leading-normal"
                style={{ color: lineColor(evt) }}
                aria-hidden
              >
                ●
              </span>
              <Text
                kind="body/regular/sm"
                className={`min-w-0 break-words ${isRunning ? 'font-semibold' : ''}`}
              >
                {formatEvent(evt)}
              </Text>
            </div>
          );
        })}
      </div>
    </div>
  );
};
