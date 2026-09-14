// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  GROUP_COLOR,
  NODES,
  type NodeGroup,
  type NodeStatus,
  type SwarmNode,
  type SwarmState,
} from '@agent-hardener/components/swarm/swarmModel';
import { ACCENT, FEEDBACK, tint } from '@agent-hardener/theme';
import { ExpandableMessage } from '@nemo/common';
import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { FC } from 'react';

// A dynamic-colored swatch via the SVG `fill` attribute (not an inline style, which lint forbids).
const Dot: FC<{ color: string }> = ({ color }) => (
  <svg width={10} height={10} aria-hidden>
    <circle cx={5} cy={5} r={5} fill={color} />
  </svg>
);

interface NodeDetailProps {
  node: SwarmNode | null;
  swarm: SwarmState;
}

const STATUS_LABEL: Record<NodeStatus, string> = {
  pending: 'Idle',
  running: 'Running',
  success: 'Succeeded',
  blocked: 'Blocked',
  failed: 'Failed',
};

const statusColorOf = (status: NodeStatus, base: string): string =>
  status === 'failed' ? '#ff3855' : status === 'blocked' ? '#ffab40' : base;

// These groups' `llm_call` events are internal judging/rating calls (e.g. an analyst prompt scoring an
// attack), not attacker↔victim content — keep them out of the node inspector.
const HIDDEN_LLM_CALL_GROUPS: NodeGroup[] = ['defender', 'validator', 'analyzer'];

const SectionLabel: FC<{ children: string }> = ({ children }) => (
  <Text kind="body/semibold/sm" className="uppercase tracking-wide text-subtle">
    {children}
  </Text>
);

// Marks the victim's side of an exchange — every prompt transcript is `selected agent ↔ victim`, so the
// response is the victim under test. Tinted the victim's swarm color to match the graph.
const VictimTag: FC = () => (
  <span
    className="rounded px-2 py-1 text-xs font-semibold uppercase tracking-wide"
    style={{ color: ACCENT.blue, backgroundColor: tint(ACCENT.blue, 15) }}
  >
    Victim
  </span>
);

// A manager isn't an agent — it has no prompts of its own. Clicking it rolls up its swarm's agents instead.
const ManagerRollup: FC<{ node: SwarmNode; swarm: SwarmState }> = ({ node, swarm }) => {
  const agents = NODES.filter((n) => n.group === node.group && !n.isManager);
  const totalExchanges = agents.reduce(
    (sum, a) => sum + (swarm.nodeExchanges[a.id]?.length ?? 0),
    0
  );
  return (
    <Stack gap="density-xs">
      <SectionLabel>Swarm</SectionLabel>
      {agents.map((agent) => {
        const agentStatus = swarm.statuses[agent.id] ?? 'pending';
        const count = swarm.nodeExchanges[agent.id]?.length ?? 0;
        return (
          <Flex key={agent.id} className="items-center justify-between">
            <Flex className="items-center gap-2">
              <Dot color={statusColorOf(agentStatus, GROUP_COLOR[agent.group])} />
              <Text kind="body/regular/sm">{agent.title}</Text>
            </Flex>
            <Text kind="body/regular/sm" className="text-subtle">
              {STATUS_LABEL[agentStatus]}
              {count ? ` · ${count} prompt${count === 1 ? '' : 's'}` : ''}
            </Text>
          </Flex>
        );
      })}
      <Text kind="body/regular/sm" className="text-subtle">
        {agents.length} agent{agents.length === 1 ? '' : 's'} · {totalExchanges} exchange
        {totalExchanges === 1 ? '' : 's'}
      </Text>
    </Stack>
  );
};

// Inspector for the node selected in the swarm graph: role + status, then either an agent's activity log +
// prompt<->response transcript, or a manager's swarm roll-up.
export const NodeDetail: FC<NodeDetailProps> = ({ node, swarm }) => {
  if (!node) {
    return (
      <Text kind="body/regular/md" className="text-subtle">
        Select an agent in the graph to inspect its activity, logs, and prompts.
      </Text>
    );
  }
  const color = GROUP_COLOR[node.group];
  const status = swarm.statuses[node.id] ?? 'pending';
  const logs = swarm.nodeLogs[node.id] ?? [];
  const exchanges = swarm.nodeExchanges[node.id] ?? [];
  const llmCalls = HIDDEN_LLM_CALL_GROUPS.includes(node.group)
    ? []
    : (swarm.nodeLlmCalls[node.id] ?? []);
  const header = (
    <Stack gap="density-sm">
      <Text kind="body/semibold/lg">{node.title}</Text>
      <Flex className="items-center gap-2">
        <Dot color={color} />
        <Text kind="body/regular/sm">
          {node.group}
          {node.isManager ? ' · manager' : ''}
        </Text>
      </Flex>
      <Flex className="items-center gap-2">
        <Dot color={statusColorOf(status, color)} />
        <Text kind="body/regular/md">{STATUS_LABEL[status]}</Text>
      </Flex>
    </Stack>
  );

  if (node.isManager) {
    return (
      <Stack gap="density-lg" className="min-h-0">
        {header}
        <ManagerRollup node={node} swarm={swarm} />
      </Stack>
    );
  }

  return (
    <Stack gap="density-lg" className="min-h-0">
      {header}

      <Stack gap="density-xs">
        <SectionLabel>Activity</SectionLabel>
        {logs.length === 0 ? (
          <Text kind="body/regular/sm" className="text-subtle">
            No activity yet.
          </Text>
        ) : (
          <Stack gap="density-xs" className="font-mono">
            {logs.map((log, index) => (
              <Text
                key={index}
                kind="body/regular/sm"
                style={log.level === 'error' ? { color: FEEDBACK.danger } : undefined}
              >
                <span className="text-subtle">{log.label}</span>
                {log.text ? ` ${log.text}` : ''}
              </Text>
            ))}
          </Stack>
        )}
      </Stack>

      <Stack gap="density-xs">
        <SectionLabel>{`Prompts (${exchanges.length})`}</SectionLabel>
        {exchanges.length === 0 ? (
          <Text kind="body/regular/sm" className="text-subtle">
            No prompts yet — they appear as this agent runs.
          </Text>
        ) : (
          <Stack gap="density-sm">
            {exchanges.map((exchange, index) => (
              <Stack key={index} gap="density-xs" className="rounded-md border border-base p-2">
                <Flex className="items-center justify-between">
                  {exchange.label ? (
                    <Text
                      kind="body/semibold/sm"
                      className={exchange.ok ? 'text-subtle' : undefined}
                      style={exchange.ok ? undefined : { color: FEEDBACK.danger }}
                    >
                      {exchange.label}
                    </Text>
                  ) : (
                    <span />
                  )}
                  {exchange.blocked !== undefined ? (
                    <Badge color={exchange.blocked ? 'red' : 'green'}>
                      {exchange.blocked ? 'Blocked' : 'Allowed'}
                    </Badge>
                  ) : null}
                </Flex>
                <SectionLabel>Request</SectionLabel>
                <ExpandableMessage message={exchange.request || '(empty)'} characterLimit={220} />
                <Flex className="items-center gap-2">
                  <SectionLabel>Response</SectionLabel>
                  <VictimTag />
                </Flex>
                <ExpandableMessage message={exchange.response || '(empty)'} characterLimit={220} />
              </Stack>
            ))}
          </Stack>
        )}
      </Stack>

      {llmCalls.length > 0 ? (
        <Stack gap="density-xs">
          <SectionLabel>{`LLM calls (${llmCalls.length})`}</SectionLabel>
          <Stack gap="density-sm">
            {llmCalls.map((call, index) => (
              <Stack key={index} gap="density-xs" className="rounded-md border border-base p-2">
                {call.label ? (
                  <Text kind="body/regular/sm" className="text-subtle">
                    {call.label}
                  </Text>
                ) : null}
                <SectionLabel>Prompt</SectionLabel>
                <ExpandableMessage message={call.request || '(empty)'} characterLimit={220} />
                <SectionLabel>Completion</SectionLabel>
                <ExpandableMessage message={call.response || '(empty)'} characterLimit={220} />
              </Stack>
            ))}
          </Stack>
        </Stack>
      ) : null}
    </Stack>
  );
};
